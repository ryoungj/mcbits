import torch
import numpy as np
import random
import time
import os

import torch.utils.data
from torch import nn

from mcbits.argsparser import str2bool, get_train_parser, get_compress_parser, get_train_args, get_compress_args
from mcbits import coders
from mcbits import util
from utils import set_gpu, get_dataset, get_model, set_seed, torch_fun_to_numpy_fun

nat2bit = np.log2(np.e) / 784.


def compute_iterative_post_params(data_loader, model, num_steps=1000, lr=0.005, num_particles=1, bound="ELBO"):
    """Pre-compute posterior params by amortized-iterative inference.

    Amortized-iterative inference initializes the posterior parameters from the encoder prediction (amortized inference)
    and then optimize the variational objective w.r.t. to the posterior parameters (iterative inference). See details
    in https://arxiv.org/pdf/2006.04240.pdf.
    """
    device = next(model.parameters()).device
    for param in model.parameters():
        param.requires_grad = False

    post_params_all = []

    for idx, (data, _) in enumerate(data_loader):
        data = data.to(device)
        with torch.no_grad():
            init_post_params = model.encode(data)
        post_params = [nn.Parameter(post_param.data).to(device) for post_param in init_post_params]

        # we need to re-initilize ADAM optimizer for each batch (i.e., each optimization)
        optimizer = torch.optim.Adam(post_params, lr=lr)

        for i in range(num_steps):
            optimizer.zero_grad()
            loss = model.loss(data, bound=bound, num_particles=num_particles, latent_params=post_params)
            loss *= post_params[0].shape[0]
            loss.backward()
            optimizer.step()

        post_params_all.append([post_param.data.cpu().numpy() for post_param in post_params])

    post_params_all = list(zip(*post_params_all))
    post_params_all = [np.concatenate(post_params_list, axis=0) for post_params_list in post_params_all]

    return post_params_all


def main_worker(args, train_args):
    # LOAD MODEL
    latent_dim = train_args.latent_dim
    model = get_model(train_args)
    model = set_gpu(args, model)
    model.load_state_dict(torch.load(train_args.ckpt_path, map_location=torch.device('cpu')))
    model.eval()

    # SPECIFY MESSAGE
    batch_size = args.batch_size  # specified test batch size
    assert args.num_compress % batch_size == 0
    args.batch_size = args.num_compress
    data = get_dataset(args)
    images = next(iter(data.test_loader))[0]  # load `num_compress` samples
    # since we use dynamically binarized dataset, to ensure consistency, we need to first load the images
    # (which have been binarized using a fix seed), then construct the dataloader later for computing
    # the ideal bitrates and iterative inference
    test_dataset = torch.utils.data.TensorDataset(images, torch.empty((images.shape[0],)))
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    message = [image.float().view(1, -1) for image in images]  # construct the message
    dimension = message[0].shape[1]

    # PRE-COMPUTE ITERATIVE INFERENCE PARAMS
    if args.iterative_post_improvement:
        post_params_all = compute_iterative_post_params(test_loader, model,
                                                        num_steps=args.iterative_improvement_steps,
                                                        lr=args.iterative_improvement_lr,
                                                        num_particles=args.num_particles,
                                                        bound=train_args.bound)
        global index, is_encode  # global variables used to select the precomputed params
        index = 0
        is_encode = True

    # SPECIFY CODER
    # STEP 1: turn model to numpy function
    rec_net = torch_fun_to_numpy_fun(model.encode, device=args.gpu)
    gen_net = torch_fun_to_numpy_fun(model.decode, device=args.gpu)

    # STEP 2: quantize latent and observation distributions
    # the proposal/posterior distributions are discretized to `buckets` which have equal mass under the prior
    # see https://arxiv.org/pdf/1901.04866.pdf for details.
    def get_prior_count_stat_func():
        """Compute the discretized prior distribution.

        The prior distribution is discretized to 2^log_num_bucket buckets where each bucket has equal probability mass
        (1 / 2^log_num_bucket) under the prior.

        Returns:
            count_stat_func: the statistical function of the discrete distribution
        """
        num_prior_buckets = (1 << args.log_num_bucket)
        counts = np.ones((latent_dim, num_prior_buckets)) / num_prior_buckets
        count_stat_func = util.CategoricalMulti(latent_dim, args.prior_mprec, counts)
        return count_stat_func

    def get_cond_count_stat_func(z):
        """Compute the discretized conditional likelihood distribution.

        This function takes the latent bucket indices as input and compute the statistical function of the discretized
        conditional distribution. The function can compute for a single or multiple particles in a batch. For batched
        computation, `z` is a list/array of `z`s and are computed in a batch. This might lead to decode check f
        ailure since the batched computation results (at encoding time) may be slightly different from the unbatched
        ones computed individually (at decoding time) in PyTorch due to non-determinism.

        Args:
            z: the latent bucket indices in the range [0, 2^log_num_bucket), can be a single or batched particles

        Returns:
            count_stat_funcs: a list or a single statistical function of the discrete distribution
        """
        # FIXME: Note that due to numerical error, the batch output is not exactly equal to the single outputs, which makes
        # FIXME: decode/encode mimatch. We might fix it by lowering the obs_mprec

        # convert the bucket indices to latent values
        # for each bucket, the bucket mass center is computed as its representative latent
        y = util.std_gaussian_centres(args.log_num_bucket)[np.array(z)].astype(np.float32)
        obs_params = gen_net(np.reshape(y, (-1, latent_dim)))
        bs = obs_params.shape[0]
        count_stat_funcs = []
        for i in range(bs):
            counts = model.obs_model.counts(obs_params[i])
            count_stat_func_i = util.CategoricalMulti(dimension, args.cond_mprec, counts)
            count_stat_funcs.append(count_stat_func_i)

        batch = isinstance(z, list) and bs > 1
        if not batch:
            return count_stat_funcs[0]
        else:
            return count_stat_funcs

    def get_prop_count_stat_func(x):
        """Compute the discretized posterior/proposal distribution.

        This function takes the observation as input and compute the statistical function of the discretized posterior
        distribution. The posterior distribution is discretized with the same set of buckets as the prior dsitrbution.


        Args:
            x: the observation data
            index: the data index, used when iterative inference is applied to select the pre-computed posterior params

        Returns:
            count_stat_func: the statistical function of the discrete distribution
        """
        if isinstance(x, torch.Tensor):
            x = x.numpy()
        x = x.reshape(1, dimension).astype(np.float32)
        post_mean, post_stdd = rec_net(x)

        if args.iterative_post_improvement:
            global index, is_encode
            post_mean, post_stdd = post_params_all[0][index], post_params_all[1][index]
            if is_encode:
                index += 1
            else:
                index -= 1

        post_mean, post_stdd = np.ravel(post_mean), np.ravel(post_stdd)
        count_stat_func = util.DiscreteGaussianMulti(latent_dim, args.prop_mprec, args.log_num_bucket, post_mean,
                                                     post_stdd)
        return count_stat_func

    # STEP 3: initialize coder
    coders_supported = ["StochasticCoder", "BitsBackCoder", "ISBitsBackCoder", "CISBitsBackCoder"]
    assert args.coder in coders_supported, f"{args.coder} is specified, " \
                                           f"but only coders in {coders_supported} are supported for this experiment!"
    coder_kwargs = {
        "mprec": 10,  # default mprec
        "lprec": args.lprec,
        "bprec": args.bprec,
        "use_statfunc": True,
        "multidim": True,
        "get_prior_count_stat_func": get_prior_count_stat_func,
        "get_cond_count_stat_func": get_cond_count_stat_func,
        "get_prop_count_stat_func": get_prop_count_stat_func,
    }
    if args.coder in ["StochasticCoder", "BitsBackCoder"]:
        assert args.num_particles == 1
    if args.coder == "ISBitsBackCoder":
        coder_kwargs["num_particles"] = args.num_particles
        coder_kwargs["batch_compute"] = args.batch_compute
    if args.coder == "CISBitsBackCoder":
        coder_kwargs["num_particles"] = args.num_particles
        coder_kwargs["batch_compute"] = args.batch_compute

        # for cis, we use shift sampling operators to generate other latents
        def shift_sampling(shift, precision):
            assert np.all(0.0 <= shift) and np.all(shift <= 1.0)
            upper = 1 << precision
            shift_scaled = (shift * upper).astype(int)

            def operator(u, prop_count_stat_func):
                if not isinstance(u, np.ndarray):
                    u = np.array(u)
                return (u + shift_scaled) % upper

            def inverse_operator(u, prop_count_stat_func):
                if not isinstance(u, np.ndarray):
                    u = np.array(u)
                return (u - shift_scaled) % upper

            return (operator, inverse_operator)

        sampling_shifts = [np.random.rand(latent_dim, ) for _ in range(args.num_particles - 1)]
        sampling_shifts.insert(0, np.zeros(latent_dim, ))
        shift_sampling_operators = [shift_sampling(shift, args.prop_mprec) for shift in sampling_shifts]

        coder_kwargs["bijective_operators"] = shift_sampling_operators

    coder = coders.__dict__[args.coder](**coder_kwargs)

    # COMPUTE IDEAL RATES
    print("=> Computing ideal bitrate...")
    avg = lambda x: torch.mean(torch.cat(x))

    if args.coder == "CISBitsBackCoder":
        iwae_all = []
        with torch.no_grad():
            for idx, (data, _) in enumerate(test_loader):
                data = data.to(args.gpu)
                post_params = [torch.Tensor(post_params_all[0][idx * batch_size:(idx + 1) * batch_size]).to(args.gpu),
                               torch.Tensor(post_params_all[1][idx * batch_size:(idx + 1) * batch_size]).to(args.gpu)] \
                    if args.iterative_post_improvement else None
                _, iwae, _ = model.compute_bounds(data, num_particles=args.num_particles, latent_params=post_params,
                                                  coupled_sampling=True, sampling_shifts=sampling_shifts)
                iwae_all.append(nat2bit * iwae)

        ideal_bitrate = -avg(iwae_all)
    else:
        elbo_all, iwae_all, joint_all = [], [], []
        with torch.no_grad():
            for idx, (data, _) in enumerate(test_loader):
                data = data.to(args.gpu)
                post_params = [torch.Tensor(post_params_all[0][idx * batch_size:(idx + 1) * batch_size]).to(args.gpu),
                               torch.Tensor(post_params_all[1][idx * batch_size:(idx + 1) * batch_size]).to(args.gpu)] \
                    if args.iterative_post_improvement else None
                elbo, iwae, joint = model.compute_bounds(data, num_particles=args.num_particles,
                                                         latent_params=post_params)
                elbo_all.append(nat2bit * elbo)
                iwae_all.append(nat2bit * iwae)
                joint_all.append(nat2bit * joint)

        if args.coder == "StochasticCoder":
            ideal_bitrate = -avg(joint_all)
        if args.coder == "BitsBackCoder":
            ideal_bitrate = -avg(elbo_all)
        if args.coder == "ISBitsBackCoder":
            ideal_bitrate = -avg(iwae_all)

    # ENCODE MESSAGE
    print("=> Encoding messages...")
    encode_start_time = time.time()
    coder.encode(message, print_progress=True)

    # PRINT RESULT
    message_length = len(message)
    print("=> Encoded {} images in {:.2f}s".format(message_length, time.time() - encode_start_time))
    print("Ideal bit length:")
    print("\t{:.4f} bits/sym".format(ideal_bitrate * dimension))
    print("\t{:.4f} bits/dim".format(ideal_bitrate))
    print("Net bit length:")
    print("\t{:.3f} bits".format(coder.net_bit_length))
    print("\t{:.4f} bits/sym".format(coder.net_bit_length / message_length))
    print("\t{:.4f} bits/dim".format(coder.net_bit_length / message_length / dimension))
    print("Total bit length:")
    print("\t{:.3f} bits".format(coder.bit_length))
    print("\t{:.4f} bits/sym".format(coder.bit_length / message_length))
    print("\t{:.4f} bits/dim".format(coder.bit_length / message_length / dimension))

    # DECODE MESSAGE
    if args.decode_check:
        if args.iterative_post_improvement:
            is_encode = False
            index -= 1
        decode_start_time = time.time()
        dec_message = coder.decode(len(message), print_progress=False)
        print("=> Decoded {} images in {:.2f}s.\n=> Decode check successful!".format(message_length,
                                                                                     time.time() - decode_start_time))
        assert np.allclose(np.concatenate(message), np.stack(dec_message)), \
            "Decoded message does not match encoded message"


def main():
    np.seterr(all='raise')

    compress_parser = get_compress_parser()

    # add specific arguments
    compress_parser.add_argument(
        "--iterative_post_improvement",
        type=str2bool,
        default=False,
        help="whether iteratively improve the posterior parameter prediction",
    )
    compress_parser.add_argument(
        "--iterative_improvement_steps",
        default=2000,
        type=int,
        help="number of steps the optimizer takes to improve the posterior "
             "initial params coming from the encoder.",
    )
    compress_parser.add_argument(
        "--iterative_improvement_lr",
        default=0.005,
        type=float,
        help="the optimizer learning rate to improve the posterior params coming from the encoder.",
    )

    compress_args = get_compress_args(compress_parser)
    print("=> Arguments:", compress_args)

    set_seed(compress_args.seed, deterministic=True)

    # get train args from saved train config file
    train_parser = get_train_parser()
    train_args = get_train_args(train_parser, config=compress_args.train_config)

    main_worker(compress_args, train_args)


if __name__ == '__main__':
    main()
