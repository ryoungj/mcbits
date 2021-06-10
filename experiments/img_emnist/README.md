# Lossless image compression on EMNIST dataset

Here we include the code for reproducing the experiments of lossless image compression on the dynamically binarized EMNIST dataset.

## Usage

The overall pipeline consists of:
- training VAE models on the (training) dataset
- evaluating the ideal bitrates on the (test) dataset (i.e., the neg. variational bound of trained models)
- compress the (test) dataset with trained models


### Training VAE models
```
python train.py --config configs/train_bin_emnist.yml [ARGUMENT_LIST]
```
The basic training arguments are specified in the config file [`configs/train_bin_emnist.yml`](./configs/train_bin_emnist.yml) 
and can be overrided by the `[ARGUMENT_LIST]` (see [`mcbits/argsparser.py`](../../mcbits/argsparser.py) for the argument list). 
The key arguments for training include:
- `--exp_name`: the training experiment name used as the experimental directory path for logging and saving the model checkpoint
- `--gpu`: the GPU used for training (set to `-1` for using CPU)
- `--dataset`: the dataset used for training (see datasets in [`datasets/`](./datasets))
- `--split`: the split of dataset, exclusively used for `EMNIST` dataset (choices: `{mnist, letters}`)
- `--bound`: the variational bound for training VAEs (choices: `{ELBO, IWAE}`)
- `--num_particles`: the number of particles for training the model

After training, the training arguments are saved as a training config file `train_config.yml` in the experiment directory 
specified by `--exp_name`, which will be be reloaded when compressing. 
The model checkpoint `model.pt` is saved in the same directory.


### Evaluation & Compressing with trained models
```
python compress.py --config configs/compress_bin_emnist.yml [ARGUMENT_LIST]
```
The basic compressing arguments are specified in the config file [`configs/compress_bin_emnist.yml`](./configs/compress_bin_emnist.yml) 
and can be overrided by the `[ARGUMENT_LIST]` (see [`mcbits/argsparser.py`](../../mcbits/argsparser.py) for the argument list). 
Compressing shares some arguments with training, such as `--exp_name`, `--gpu`, `--dataset`, `--split`, etc., but are 
specified for compressing experiments. 
The key arguments for compressing include:
- `--train_config`: the saved training config (specifying the model and the checkpoint loaded for compression)
- `--num_compress`: the (maximum) number of data samples to compress
- `--batch_compute`: whether batch the computation of conditional likelihood over particles (if `True`, it might lead 
to decode check failure due to some non-deterministic computation in PyTorch
- `--decode_check`: whether run decoding and check the correctness of decoded results
- `--coder`: the coder to use for compression (see the supported coder list in `CODER_LIST` in [`mcbits/coders.py`](../../mcbits/coders.py))
- `--num_particles`: number of particles for compression (for BB-IS and BB-CIS)
- `--lprec`, `--bprec`: the precisions for the rANS stack
- `--log_num_bucket`, `--prior_mprec`, `--prop_mprec`, `--cond_mprec`: the precisions for discretizing the latent and observation distributions
- `--iterative_post_improvement`: where apply amortized iterative inference to improve posterior predictions
- `--iterative_improvement_steps`, `--iterative_improvement_lr`: the number of optimization steps and learning rate for iterative inference

The ideal bitrates (i.e., the neg. variational bound of trained models) are first evaluated on the (test) dataset. Then 
the (test) dataset is compressed with the specified coder and the trained model, the true net bitrates and the total 
bitrates (plus initial bits) are computed.

## Reproducing the results
We provide the training configs and model checkpoints in [`checkpoints/`](checkpoints) for reproducing our experiments. 
Each directory corresponds to a training run, named as `train_bin_emnist_split=[SPLIT]_num=[NUM_PARTICLES]`. By default, 
`--num_compress=500` images from the test set are compressed for each experiment. To compress the whole test set, set 
`--num_compress=10000` for the `mnist` split and `--num_compress=20800` for the `letters` split. Sometimes, the quantization 
gap (quantified by the gap between the net bitrate and the ideal bitrate) is not negligible, we recommend tuning the precision
parameters if needed. 

### In- & Out-of-Distribution Compression Performance
With the provided checkpoints, you can measure both in- & out-of-distribution compression performance of BB-IS (by modifying 
the `--split` argument) and the effect of increasing the number of particles (by using models trained with different 
particle numbers and modifying the `--num_particles` argument when compressing). We provide results below for reference, 
where each setting means `TRAIN SPLIT`&#8594;` COMPRESS SPLIT`, and **net** bitrates (bits/dim) are compared. 

| Method / Setting | MNIST &#8594; MNIST | MNIST &#8594; Letters | Letters &#8594; Letters | Letters &#8594; MNIST |
| --- | :---: | :---: | :---: | :---: |
| BB-ELBO       | 0.2382 | 0.3068 | 0.2471 | 0.2596 | 
| BB-IS (5)     | 0.2335 | 0.2880 | 0.2413 | 0.2512 |
| BB-IS (50)    | 0.2305 | 0.2784 | 0.2371 | 0.2462 |
| Savings       | 3.2% | **9.3%** | 4.0% | **5.2%** |

### Comparison with Iterative Inference
By turning setting the `--iterative_post_improvement` argument to `True`, you can also compare BB-IS with iterative inference and combine 
them together to get further improvement. We provide results below for reference, where `IF (50)` denotes `--iterative_improvement_steps=50`.

| Method / Setting | MNIST &#8594; MNIST | MNIST &#8594; Letters |
| --- | :---: | :---: | 
| BB-ELBO               | 0.2382 | 0.3068 |
| BB-ELBO-IF (50)       | 0.2351 | 0.2915 |  
| BB-IS (50)            | 0.2305 | 0.2784 | 
| BB-IS (50)-IF (50)    | 0.2291 | 0.2702 |  
| Savings               | 3.8% | **11.9%** | 


### Comparison with Coupled Variant (BB-CIS)
By changing the `--coder` to `CISBitsBackCoder`, you can compare BB-IS with its coupled variant (BB-CIS), especially in terms 
of the total bitrate (including the initial bit cost). We provide results below for reference, which is done in the 
`MNIST`&#8594;`MNIST` setting, and each cell represents **net** bitrate/**total** bitrate.

| N / Method | BB-IS | BB-CIS |
| --- | :---: | :---: | 
| 1     | 0.2382 / 0.2393 | 0.2378 / 0.2400 |
| 5     | 0.2334 / 0.2382 | 0.2333 / 0.2355 |
| 50    | 0.2305 / **0.2782** | 0.2303 / **0.2325** |



