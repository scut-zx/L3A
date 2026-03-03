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
parser.add_argument('--epochs', type=int, default=1)

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

    # DDP
    if 'LOCAL_RANK' in os.environ:
        args.local_rank = int(os.environ['LOCAL_RANK'])
    if 'WORLD_SIZE' in os.environ:
        args.world_size = int(os.environ['WORLD_SIZE'])
    if 'RANK' in os.environ:
        args.rank = int(os.environ['RANK'])
    else:
        args.rank = 0
        args.world_size = 1
        args.local_rank = 0

    torch.cuda.set_device(args.local_rank)
    
    if not dist.is_initialized():
        print(f'[Process {args.rank}] Initializing distributed process group, total {args.world_size} processes.')
        dist.init_process_group(backend='nccl', init_method='env://')

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

    if dist.is_initialized():
        dist.destroy_process_group()

    end_time = time.time()
    elapsed_time = end_time - start_time
    if args.rank == 0:
        print(f"Running time: {elapsed_time:.2f} 秒")

if __name__ == '__main__':
    main()