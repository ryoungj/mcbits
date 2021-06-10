import torch
import torch.utils.data
from mcbits.argsparser import get_train_parser, get_train_args, dump_args
import random
import time
import os
from utils import *

PRINT_FREQUENCY = 50

def train(epoch, args, model, data_loader, optimizer):
    model.train()
    losses = []
    start = time.time()
    for batch_idx, (data, _) in enumerate(data_loader):
        data = data.to(args.gpu)
        optimizer.zero_grad()
        loss = model.loss(data, bound=args.bound, num_particles=args.num_particles)
        loss.backward()
        losses.append(loss.item())
        optimizer.step()
        if batch_idx % PRINT_FREQUENCY == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\r'.format(
                epoch, batch_idx * len(data), len(data_loader.dataset),
                       100. * batch_idx / len(data_loader),
                loss.item()), end="")

    print('====> Epoch: {} Average loss: {:.4f} lr: {:.5f} time: {:.2f} s/epoch'.format(
        epoch, np.mean(losses), optimizer.param_groups[0]['lr'], time.time() - start))


def test(epoch, args, model, data_loader):
    model.eval()
    losses = []
    for data, _ in data_loader:
        data = data.to(args.gpu)
        loss = model.loss(data, bound=args.bound, num_particles=args.num_particles)
        losses.append(loss.item())
    test_loss = np.mean(losses)
    print('====> Epoch: {} Test loss: {:.6f}'.format(
        epoch, test_loss
    ))

    return test_loss


def main_worker(args):
    # build model, optimizer and dataset
    data = get_dataset(args)
    model = get_model(args)
    model = init_model(args, model, data)
    model = set_gpu(args, model)
    optimizer = get_optimizer(args, model)
    scheduler = get_scheduler(args, optimizer)

    min_test_loss = np.inf
    for epoch in range(1, args.epochs + 1):
        train(epoch, args, model, data.train_loader, optimizer)
        test_loss = test(epoch, args, model, data.test_loader)
        # model.reconstruct(data.recon_dataset)
        # model.sample(args.gpu)
        scheduler.step(epoch)

        if test_loss < min_test_loss:
            min_test_loss = test_loss
            torch.save(model.state_dict(), args.ckpt_path)
            print("====> Save checkpoint to '{}' (epoch {})".format(args.ckpt_path, epoch))
        print("\n")
    torch.save(model.state_dict(), args.ckpt_path)
    print("====> Save checkpoint to '{}' (epoch {})".format(args.ckpt_path, epoch))
    print(f"\n=> Finish training (epoch {args.epochs})!")


def main():
    train_parser = get_train_parser()
    args = get_train_args(train_parser)
    assert args.binarize is True and args.model == "BinaryVAE"
    print("=> Arguments:", args)

    set_seed(args.seed)

    os.makedirs(args.expdir, exist_ok=True)
    dump_args(args, args.config_path)
    main_worker(args)


if __name__ == '__main__':
    main()
