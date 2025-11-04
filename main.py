from MultiLabelIncremental_L3A import MultiLabelIncremental
import torch.distributed as dist
import argparse
import yaml
import torch
import os 
import random
import numpy as np
import time

parser = argparse.ArgumentParser(description='Start Training')
parser.add_argument('--local_rank', default=0, type=int, help='local rank for DistributedDataParallel')
parser.add_argument('--options', nargs='*')
parser.add_argument('--output_name', type=str)
parser.add_argument('--Hidden', type=int, default=4096)
parser.add_argument('--rg', type=float, default=1000)
parser.add_argument('--base_classes', type=int, default=40)
parser.add_argument('--task_size', type=int, default=10)
parser.add_argument('--thre', type=float, default=0.7)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_options(args, options):
    for o in options:
        with open(o) as f:
            config = yaml.safe_load(f)
            for k, v in config.items():
                if not hasattr(args, k) or getattr(args, k) is None:
                    setattr(args, k, v)

def main():
    start_time = time.time()
    args = parser.parse_args()
    if args.options:
        load_options(args, args.options)

    args.rank = 0
    args.world_size = 0
    # Distributed
    # if 'WORLD_SIZE' in os.environ:
    #     # args.rank = int(os.environ["RANK"])
    #     args.rank = 0
    #     args.world_size = int(os.environ['WORLD_SIZE'])
    # print('Training in distributed mode with multiple processes, 1 GPU per process. Process %d, total %d.'
    #       % (args.rank, args.world_size))

    torch.cuda.set_device(args.local_rank)
    os.environ['MASTER_ADDR'] = 'localhost'

    os.environ['MASTER_PORT'] = '4433'

    dist.init_process_group(backend='nccl', init_method='env://', rank = 0, world_size = 1)

    args.logger_dir = 'logs/' + args.output_name 
    args.tensorboard_dir = 'tensorboard/' + args.output_name 
    args.model_save_path = 'saved_models/' + args.output_name 

    set_seed(2025)

    if args.arch == 'l3a':
        multi_incremental = MultiLabelIncremental(args)
        multi_incremental.train()
    else:
        print('error')
    del multi_incremental

    end_time = time.time()
    elapsed_time = end_time - start_time
    print("Running time (in seconds):", elapsed_time)


if __name__ == '__main__':
    main()
