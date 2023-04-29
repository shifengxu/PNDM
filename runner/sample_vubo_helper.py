import os

from runner.sample_runner import SampleRunner
from schedule.schedule_batch import ScheduleBatch
from utils import log_info

class ScheduleSampleConfig:
    def __init__(self, lp=None, calo=None):
        """
        :param lp: learning portion
        :param calo: calculating loss order. order when calculating loss during schedule
        """
        self.lp = lp
        self.calo = calo

    def parse(self, cfg_str):
        # cfg_str is like: "0.1   : 0"
        arr = cfg_str.strip().split(':')
        if len(arr) != 2: raise ValueError(f"Invalid cfg_str: {cfg_str}")
        self.lp             = float(arr[0].strip())
        self.calo           = int(arr[1].strip())
        return self

class ScheduleSampleResult:
    def __init__(self, ssc: ScheduleSampleConfig, key=None, fid=None, fid_std=None):
        self.ssc = ssc
        self.key = key
        self.fid = fid
        self.fid_std = fid_std
        self.notes = ''

def load_plans_from_file(f_path):
    log_info(f"load_plans_from_file(): {f_path}")
    with open(f_path, 'r') as f:
        lines = f.readlines()
    cnt_empty = 0
    cnt_comment = 0
    cnt_valid = 0
    plan_map = {}  # key is string, value is an array of ScheduleSampleConfig
    for line in lines:
        line = line.strip()
        if line == '':
            cnt_empty += 1
            continue
        if line.startswith('#'):
            cnt_comment += 1
            continue
        cnt_valid += 1
        key, cfg_str = line.strip().split(':', 1)
        key = key.strip()
        ssc = ScheduleSampleConfig().parse(cfg_str)
        if key in plan_map:
            plan_map[key].append(ssc)
        else:
            plan_map[key] = [ssc]
    log_info(f"  cnt_empty  : {cnt_empty}")
    log_info(f"  cnt_comment: {cnt_comment}")
    log_info(f"  cnt_valid  : {cnt_valid}")
    log_info(f"  cnt key    : {len(plan_map)}")
    for idx, key in enumerate(sorted(plan_map)):
        log_info(f"  {idx:03d} {key}: {len(plan_map[key])}")
    log_info(f"load_plans_from_file(): {f_path}... Done")
    if 'default' not in plan_map:
        raise ValueError(f"'default' must be in plan file: {f_path}")
    return plan_map

def output_ssr_list(ssr_list, f_path):
    log_info(f"Save file: {f_path}")
    with open(f_path, 'w') as f_ptr:
        f_ptr.write(f"#  FID    : std    : lp    : calo: key                 : notes\n")
        for ssr in ssr_list:
            ssc = ssr.ssc
            s = f"{ssr.fid:9.5f}: {ssr.fid_std:.5f}: {ssc.lp:.4f}:" \
                f" {ssc.calo:4d}: {ssr.key.ljust(20)}: {ssr.notes}\n"
            f_ptr.write(s)
        # for
    # with

class SampleVuboHelper:
    def __init__(self, args, sample_runner: SampleRunner):
        self.args = args
        self.sample_runner = sample_runner

    def schedule_sample(self):
        args = self.args
        plan_map = load_plans_from_file(args.ss_plan_file)
        log_info(f"SampleVuboHelper::schedule_sample() *********************************")
        sb = ScheduleBatch(args)
        ori_dir = args.ab_original_dir
        sum_dir = args.ab_summary_dir
        if not os.path.exists(sum_dir):
            log_info(f"  os.makedirs({sum_dir})")
            os.makedirs(sum_dir)
        run_hist_file = os.path.join(sum_dir, "ss_run_hist.txt")
        fid_best_file = os.path.join(sum_dir, "ss_run_best.txt")
        # ori_dir file name is like: ts_t-1.0_s20_rho_ab_3_3kutta.txt
        file_list = [f for f in os.listdir(ori_dir) if f.endswith('.txt')]
        file_list = [os.path.join(ori_dir, f) for f in file_list]
        file_list = [f for f in file_list if os.path.isfile(f)]
        f_cnt = len(file_list)
        log_info(f"  ab_original_dir  : {args.ab_original_dir}")
        log_info(f"  ab file cnt      : {f_cnt}")
        log_info(f"  ab_scheduled_dir : {args.ab_scheduled_dir}")
        log_info(f"  sample_output_dir: {args.sample_output_dir}")
        log_info(f"  fid_input1       : {args.fid_input1}")
        run_hist = []  # running history
        fid_best = []  # best FID for each key
        for idx, f_path in enumerate(sorted(file_list)):
            log_info(f"{idx:03d}/{f_cnt}: {f_path} ----------------------------------")
            f_name = os.path.basename(f_path)   # ts_t-1.0_s20_t_ab_3_3kutta.txt
            key = f_name.split('.txt')[0]       # ts_t-1.0_s20_t_ab_3_3kutta
            ssc_arr = plan_map.get(key, plan_map['default'])
            ssr_best = None
            for ssc in ssc_arr:
                scheduled_file = sb.schedule_single(f_path, args.lr, ssc.lp, order=ssc.calo)
                key, fid_avg, fid_std = self.sample_runner.sample_times(args.repeat_times, scheduled_file)
                ssr = ScheduleSampleResult(ssc, key, fid_avg, fid_std)
                run_hist.append(ssr)
                output_ssr_list(run_hist, run_hist_file)
                if ssr_best is None or ssr_best.fid > fid_avg:
                    ssr_best = ssr
                # if
            # for
            fid_best.append(ssr_best)
            output_ssr_list(fid_best, fid_best_file)
        # for

    def sample_scheduled(self):
        args = self.args
        plan_map = load_plans_from_file(args.ss_plan_file)
        log_info(f"SampleVuboHelper::sample_scheduled() *********************************")
        sch_dir = args.ab_scheduled_dir
        sum_dir = args.ab_summary_dir
        if not os.path.exists(sum_dir):
            log_info(f"  os.makedirs({sum_dir})")
            os.makedirs(sum_dir)
        run_hist_file = os.path.join(sum_dir, "ss_run_hist.txt")
        fid_best_file = os.path.join(sum_dir, "ss_run_best.txt")
        file_list = [f for f in os.listdir(sch_dir) if f.endswith('.txt')]
        file_list = [os.path.join(sch_dir, f) for f in file_list]
        file_list = [f for f in file_list if os.path.isfile(f)]
        f_cnt = len(file_list)
        log_info(f"  ab_scheduled_dir: {args.ab_scheduled_dir}")
        log_info(f"  ab file cnt     : {f_cnt}")
        log_info(f"  ab_summary_dir  : {args.ab_summary_dir}")
        log_info(f"  fid_input1      : {args.fid_input1}")
        run_hist = []  # running history
        fid_best = []  # best FID for each key
        for idx, f_path in enumerate(sorted(file_list)):
            log_info(f"{idx:03d}/{f_cnt}: {f_path} ----------------------------------")
            f_name = os.path.basename(f_path)   # abDetail_ts_t-1.0_s20_rho_ab_3_3kutta.txt
            tmp = f_name.split('abDetail_')[-1] # ts_t-1.0_s20_rho_ab_3_3kutta.txt
            key = tmp.split('.txt')[0]          # ts_t-1.0_s20_rho_ab_3_3kutta
            ssc_arr = plan_map.get(key, plan_map['default'])
            ssr_best = None
            for ssc in ssc_arr:
                key, fid_avg, fid_std = self.sample_runner.sample_times(args.repeat_times, f_path)
                ssr = ScheduleSampleResult(ssc, key, fid_avg, fid_std)
                run_hist.append(ssr)
                output_ssr_list(run_hist, run_hist_file)
                if ssr_best is None or ssr_best.fid > fid_avg:
                    ssr_best = ssr
                # if
            # for
            fid_best.append(ssr_best)
            output_ssr_list(fid_best, fid_best_file)
        # for

# class
