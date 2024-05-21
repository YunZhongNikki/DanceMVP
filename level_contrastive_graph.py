################################################################################################################
#   This script provide the main function to pre-train a DanceMVP model - AAAI2024
#   The code is developed based on the contrastive learning(SimCLR)method provided by Thalles Silva
#   https://github.com/sthalles/SimCLR
#   We reformat the input data as dance music and motion, change the encoders to LSTM and ST-GCN
#################################################################################################################

import argparse
import torch
import torch.backends.cudnn as cudnn
from torchvision import models
from data_aug.seg_dataset import ContrastiveLearningLevelDatasetSEG

from models.st_gcn import GraphModel
from models.lstm import LSTM
from models.mlp import MLP
import itertools

#simclr
from momu_simclr import SimCLR


model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch SimCLR')
parser.add_argument('-data', metavar='DIR', default='./datasets/dance_level/eighteen_choe_10s/',
                    help='path to dataset')
parser.add_argument('-split', metavar='DIR', default=True,
                    help='split dataset into train/test')
parser.add_argument('-n_samples', metavar='DIR', default=75,
                    help='num of samples of each class used in training(7:3)')
parser.add_argument('-pair', metavar='DIR', default=True,
                    help='create pair to train?')
parser.add_argument('-istrain', metavar='DIR', default=True,
                    help='training/testing')

parser.add_argument('-j', '--workers', default=12, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=16, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.0003, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--disable-cuda', action='store_true',
                    help='Disable CUDA')
parser.add_argument('--fp16-precision', action='store_true',
                    help='Whether or not to use 16-bit precision GPU training.')


parser.add_argument('--log-every-n-steps', default=1, type=int,
                    help='Log every n steps')
parser.add_argument('--temperature', default=0.07, type=float,
                    help='softmax temperature (default: 0.07)')
parser.add_argument('--n-views', default=2, type=int, metavar='N',
                    help='Number of views for contrastive learning training.')
parser.add_argument('--gpu-index', default=0, type=int, help='Gpu index.')

#graph args
parser.add_argument('--in_channels', default=3, type=int,
                    help='single graph input channels (3)')
parser.add_argument('--num_class', default=9, type=int,
                    help='Number of action classes for contrastive learning training.')
parser.add_argument('--edge_importance_weighting', default=True, type=bool)

parser.add_argument('--n_classes', default=9, type=int,
                    help='beginner,intermidate,exper(3)')

#lstm args
parser.add_argument('--input_dim', default=130, type=int)
parser.add_argument('--hidden_dim', default=256, type=int)
parser.add_argument('--layer_dim', default=8, type=int)
parser.add_argument('--out_dim', default=256, type=int,
                    help='feature dimension (default: 128)')

#mlp args
parser.add_argument('--mlp_in', default=512, type=int)
parser.add_argument('--mlp_out', default=256, type=int)

def main():
    args = parser.parse_args()
    # check if gpu training is available
    if not args.disable_cuda and torch.cuda.is_available():
        args.device = torch.device('cuda')
        cudnn.deterministic = True
        cudnn.benchmark = True
    else:
        args.device = torch.device('cpu')
        args.gpu_index = -1

    #split train and test
    train_dataset = ContrastiveLearningLevelDatasetSEG(args.data, args.n_samples, args.split, args.pair, args.istrain)
    train_dataset.load_level_dataset()

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, drop_last=True)

    graph_args = dict()
    graph_args['layout'] = 'mocap'
    graph_args['strategy'] = 'spatial'
    model_motion = GraphModel(args.device,graph_args=graph_args,in_channels=args.in_channels,num_class=args.num_class,
                       edge_importance_weighting=args.edge_importance_weighting)

    model_music = LSTM(args.device,args.input_dim, args.hidden_dim, args.layer_dim, args.out_dim)
    model_mlp = MLP(args.device, args.mlp_in, args.mlp_out, True)

    params = [model_motion.parameters(),model_music.parameters(), model_mlp.parameters()]
    optimizer = torch.optim.Adam(itertools.chain(*params), args.lr, weight_decay=args.weight_decay)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader), eta_min=0,
                                                           last_epoch=-1)

    #  It’s a no-op if the 'gpu_index' argument is a negative integer or None.
    with torch.cuda.device(args.gpu_index):
        simclr = SimCLR(model_mo=model_motion, model_mu=model_music,model_m=model_mlp,optimizer=optimizer, scheduler=scheduler, args=args)
        simclr.train(train_loader)


if __name__ == "__main__":
    main()
