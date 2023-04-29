# copied from main.py
# main_ss: main_schedule_sample

import argparse

import yaml
import os
import torch as th

from runner.sample_runner import SampleRunner
from utils import str2bool, log_info

def args_and_config():
    parser = argparse.ArgumentParser()

    parser.add_argument("--todo", type=str, default='sample_baseline')
    parser.add_argument("--gpu_ids", nargs='+', type=int, default=[7])
    parser.add_argument("--config", type=str, default='config/ddim_cifar10.yml')
    parser.add_argument("--repeat_times", type=int, default=1, help='run XX times to get avg FID')
    parser.add_argument("--grad_method_arr", nargs='+', type=str, default=['DDIM', 'S-PNDM', 'F-PNDM', 'PF'],
                        help="DDIM|FON|S-PNDM|F-PNDM|PF")
    parser.add_argument("--num_step_arr", nargs='+', type=int, default=[10])
    parser.add_argument("--sample_output_dir", type=str, default='generated')
    parser.add_argument("--sample_count", type=int, default=100, help="sample image count")
    parser.add_argument("--sample_batch_size", type=int, default=50, help="0 mean from config file")
    parser.add_argument("--sample_ckpt_path", type=str, default='ckpt/ddim_cifar10.ckpt')
    parser.add_argument("--fid_input1", type=str, default="cifar10-train")
    args = parser.parse_args()
    ids = args.gpu_ids
    args.device = f"cuda:{ids[0]}" if th.cuda.is_available() and ids else 'cpu'

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    return args, config

def get_model(args, config):
    device = th.device(args.device)
    model_struc = config['Model']['struc']
    model_cfg = config['Model']
    log_info(f"get_model()...")
    log_info(f"  model_struc: {model_struc}")
    log_info(f"  device     : {device}")
    if model_struc == 'DDIM':
        from model.ddim import Model
        model = Model(args, model_cfg).to(device)
    elif model_struc == 'iDDPM':
        from model.iDDPM.unet import UNetModel
        model = UNetModel(args, model_cfg).to(device)
    elif model_struc == 'PF':
        from model.scoresde.ddpm import DDPM
        model = DDPM(args, model_cfg).to(device)
    elif model_struc == 'PF_deep':
        from model.scoresde.ncsnpp import NCSNpp
        model = NCSNpp(args, model_cfg).to(device)
    else:
        raise ValueError(f"Unsupported model_struct: {model_struc}")
    m_path = args.sample_ckpt_path
    log_info(f"  load: {m_path} ...")
    model.load_state_dict(th.load(m_path, map_location=device), strict=True)
    log_info(f"  load: {m_path} ...Done")
    if len(args.gpu_ids) > 1:
        log_info(f"  torch.nn.DataParallel(model, device_ids={args.gpu_ids})")
        model = th.nn.DataParallel(model, device_ids=args.gpu_ids)
    model.eval()
    return model

def main():
    args, config = args_and_config()
    log_info(f"pid : {os.getpid()}")
    log_info(f"cwd : {os.getcwd()}")
    log_info(f"args: {args}")
    log_info(f"config    : {args.config}")
    log_info(f"ckpt_path : {args.sample_ckpt_path}")

    # if args.runner == 'sample' and config['Sample']['mpi4py']:
    #     from mpi4py import MPI
    #
    #     comm = MPI.COMM_WORLD
    #     mpi_rank = comm.Get_rank()
    #     os.environ['CUDA_VISIBLE_DEVICES'] = str(mpi_rank)

    model = get_model(args, config)
    runner = SampleRunner(args, config, model)
    if args.todo == 'sample_baseline':
        runner.sample_baseline()

if __name__ == "__main__":
    main()
