import os
import copy
import yaml
import torch
import random
import argparse
import traceback

import numpy as np
import torch.multiprocessing as mp
import torch.distributed as dist

from utils import dict2namespace, namespace2dict, get_runner
from runners.BiBBDMRunner import BiBBDMRunner
from datasets.custom import CustomAlignedDataset


def parse_args_and_config():
    parser = argparse.ArgumentParser(description=globals()['__doc__'])

    # common settings
    parser.add_argument('-c', '--config', type=str, default='BB_base.yml', help='Path to the config file')
    parser.add_argument('-s', '--seed', type=int, default=1234, help='Random seed')
    parser.add_argument('-r', '--result_path', type=str, default='results', help="The directory to save results")

    parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids, 0,1,2,3 cpu=-1')
    parser.add_argument('--port', type=str, default='12355', help='DDP master port')

    parser.add_argument('--resume_model', type=str, default=None, help='model checkpoint')
    parser.add_argument('--resume_optim', type=str, default=None, help='optimizer checkpoint')

    # training settings
    parser.add_argument('-t', '--train', action='store_true', default=False, help='train the model')
    parser.add_argument('--debug', action='store_true', default=False, help='debug setting: sample at start, etc.')

    parser.add_argument('--max_epochs', type=int, default=None, help='maximum training epochs, training will end depending on which will come first (max_iters and max_epochs)')
    parser.add_argument('--max_iters', type=int, default=None, help='maximum training iterations, training will end depending on which will come first (max_iters and max_epochs)')

    # testing settings
    parser.add_argument('--sample_to_eval', action='store_true', default=False, help='sample for evaluation')
    parser.add_argument('--sample_mid_steps', action='store_true', default=False, help='save middle step results of sampling process')
    parser.add_argument('--sample_a2b', action='store_true', default=False, help='whether to sample A to B direction')
    parser.add_argument('--sample_b2a', action='store_true', default=False, help='whether to sample B to A direction')

    parser.add_argument('--sample_num', type=int, default=None, help='sample number for calculating diversity')
    parser.add_argument('--sample_step', type=int, default=None, help='number of sampling step')
    parser.add_argument('--sample_type', type=int, default=None, help='sample type')
    
    args = parser.parse_args()
    with open(args.config, 'r') as f:
        dict_config = yaml.load(f, Loader=yaml.FullLoader)

    namespace_config = dict2namespace(dict_config)
    namespace_config.args = args

    # replace config file options with args options
    if args.resume_model is not None:
        namespace_config.model.model_load_path = args.resume_model
    if args.resume_optim is not None:
        namespace_config.model.optim_sche_load_path = args.resume_optim
    if args.sample_type is not None:
        sample_types = ['uniform', 'nonuniform', 'Markovian']
        assert args.sample_type in sample_types, f'sample_type should be in {sample_types}'
        namespace_config.model.sample_type = args.sample_type
    if args.sample_step is not None:
        namespace_config.model.sample_step = args.sample_step
    if args.max_epochs is not None:
        namespace_config.training.max_epochs = args.max_epochs
    if args.max_iters is not None:
        namespace_config.training.max_iters = args.max_iters
    if args.sample_num is not None:
        namespace_config.testing.sample_num = args.sample_num
    
    if args.debug:
        namespace_config.training.max_epoches = 10
        namespace_config.training.max_iters = 10000
        namespace_config.training.save_epoch_interval = 1
        namespace_config.training.sample_iter_inverval = 200
        namespace_config.training.sample_epoch_inverval = 1
        namespace_config.training.validation_iter_interval = 200
        namespace_config.training.validation_epoch_interval = 1
        namespace_config.accumulate_grad_batches = 1

    dict_config = namespace2dict(namespace_config)

    return namespace_config, dict_config


def set_random_seed(SEED=1234):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def CPU_singleGPU_launcher(config):
    set_random_seed(config.args.seed)
    runner = get_runner(config.runner, config)
    if config.args.train:
        runner.train()
    else:
        runner.test()
    return


def DDP_launcher(world_size, run_fn, config):
    mp.spawn(run_fn,
             args=(world_size, copy.deepcopy(config)),
             nprocs=world_size,
             join=True)
    

def DDP_run_fn(rank, world_size, config):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = config.args.port
    dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)

    try:
        set_random_seed(config.args.seed)

        local_rank = dist.get_rank()
        # torch.cuda.set_device(local_rank)
        config.training.device = torch.device("cuda:%d" % local_rank)
        print('using device:', config.training.device)
        config.training.local_rank = local_rank
        runner = get_runner(config.runner, config)
        if config.args.train:
            runner.train()
        else:
            with torch.no_grad():
                runner.test()
    except Exception as e:
        print('str(Exception):\t', str(Exception))
        print('str(e):\t\t', str(e))
        print('repr(e):\t', repr(e))
        print('traceback.print_exc():')
        traceback.print_exc()
        print('traceback.format_exc():\n%s' % traceback.format_exc())
    finally:
        dist.destroy_process_group()

    return


def main():
    nconfig, dconfig = parse_args_and_config()
    args = nconfig.args

    gpu_ids = args.gpu_ids

    if gpu_ids == "-1": # Use CPU
        nconfig.training.use_DDP = False
        nconfig.training.device = torch.device("cpu")
        CPU_singleGPU_launcher(nconfig)
    else:
        gpu_list = gpu_ids.split(",")
        if len(gpu_list) > 1:
            os.environ['CUDA_VISIBLE_DEVICES'] = gpu_ids
            nconfig.training.use_DDP = True
            DDP_launcher(world_size=len(gpu_list), run_fn=DDP_run_fn, config=nconfig)
        else:
            nconfig.training.use_DDP = False
            nconfig.training.device = torch.device(f"cuda:{gpu_list[0]}")
            CPU_singleGPU_launcher(nconfig)
    
    return

if __name__ == "__main__":
    main()