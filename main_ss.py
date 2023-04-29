# copied from main.py
# main_ss: main_schedule_sample

import argparse
import yaml
import os
import torch as th

from runner.sample_runner import SampleRunner
from runner.sample_vubo_helper import SampleVuboHelper
from schedule.schedule_batch import ScheduleBatch
from utils import log_info

def args_and_config():
    parser = argparse.ArgumentParser()

    parser.add_argument("--todo", type=str, default='schedule_sample')
    # parser.add_argument("--todo", type=str, default='sample_baseline')
    parser.add_argument("--gpu_ids", nargs='+', type=int, default=[7])
    parser.add_argument("--config", type=str, default='config/ddim_cifar10.yml')
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--repeat_times", type=int, default=1, help='run XX times to get avg FID')
    parser.add_argument("--ab_original_dir", type=str,  default='./output1_cifar10/phase1_ab_original')
    parser.add_argument("--ab_scheduled_dir", type=str, default='./output1_cifar10/phase2_ab_scheduled')
    parser.add_argument("--ab_summary_dir", type=str,   default='./output1_cifar10/phase3_ab_summary')
    parser.add_argument("--ss_plan_file", type=str,     default="./output1_cifar10/vubo_ss_plan.txt")

    parser.add_argument("--grad_method_arr", nargs='+', type=str, default=['DDIM', 'F-PNDM', 'S-PNDM'],
                        help="DDIM|FON|S-PNDM|F-PNDM|PF")
    parser.add_argument("--num_step_arr", nargs='+', type=int, default=[15])

    parser.add_argument("--sample_output_dir", type=str, default='./output1_cifar10/generated')
    parser.add_argument("--sample_count", type=int, default=10, help="sample image count")
    parser.add_argument("--sample_batch_size", type=int, default=5, help="0 mean from config file")
    parser.add_argument("--sample_ckpt_path", type=str, default='ckpt/ddim_cifar10.ckpt')
    parser.add_argument("--fid_input1", type=str, default="cifar10-train")

    # arguments for schedule_batch
    parser.add_argument("--n_epochs", type=int, default=1000)
    parser.add_argument("--log_interval", type=int, default=200)
    parser.add_argument('--lr', type=float, default=0.000001)
    parser.add_argument('--lp', type=float, default=0.01, help='learning_portion')
    parser.add_argument('--aa_low', type=float, default=0.0001, help="Alpha Accum lower bound")
    parser.add_argument("--aa_low_lambda", type=float, default=10000000)
    parser.add_argument("--weight_file", type=str, default='./output1_cifar10/res_mse_avg_list.txt')
    parser.add_argument("--weight_power", type=float, default=1.0, help='change the weight value')
    parser.add_argument("--beta_schedule", type=str, default="linear")

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
    a = args.todo
    if a == 'sample_baseline':
        log_info(f"{a} ======================================================================")
        runner.sample_baseline()
    elif a == 'alpha_bar_all':
        log_info(f"{a} ======================================================================")
        runner.alpha_bar_all()
    elif a == 'schedule' or a == 'schedule_only':
        log_info(f"{a} ======================================================================")
        sb = ScheduleBatch(args)
        sb.schedule_batch()
    elif a == 'sample_scheduled':
        log_info(f"{a} ======================================================================")
        helper = SampleVuboHelper(args, runner)
        helper.sample_scheduled()
    elif a == 'schedule_sample':
        log_info(f"{a} ======================================================================")
        helper = SampleVuboHelper(args, runner)
        helper.schedule_sample()
    else:
        raise ValueError(f"Unknown todo: {a}")

if __name__ == "__main__":
    main()
