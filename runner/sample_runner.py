# copied from runner.py
import datetime
import os

import numpy as np
import torch as th
import torch_fidelity
import torchvision.utils as tvu
from scipy import integrate
from utils import log_info
import runner.method as method
from dataset import inverse_data_transform
from runner.schedule import Schedule


class SampleRunner(object):
    def __init__(self, args, config, model):
        self.args = args
        self.config = config
        self.model = model
        self.diffusion_step = config['Schedule']['diffusion_step']
        self.num_step_arr = args.num_step_arr
        self.grad_method_arr = args.grad_method_arr
        self.num_step = self.num_step_arr[0]
        self.grad_method = self.grad_method_arr[0]
        self.device = th.device(args.device)
        self.schedule = Schedule(args, config['Schedule'], self.grad_method)
        self.real_seq = None
        log_info(f"SampleRunner()...")
        log_info(f"  num_step_arr   : {self.num_step_arr}")
        log_info(f"  grad_method_arr: {self.grad_method_arr}")
        log_info(f"  device         : {self.device}")
        log_info(f"  schedule.config:")
        log_info(f"    type         : {config['Schedule']['type']}")
        log_info(f"    beta_start   : {config['Schedule']['beta_start']}")
        log_info(f"    beta_end     : {config['Schedule']['beta_end']}")
        log_info(f"    diffusion_step:{config['Schedule']['diffusion_step']}")
        log_info(f"  schedule.method: {self.grad_method}")

    def sample_baseline(self):
        def save_result(_msg_arr, _fid_arr):
            with open('./sample_all_result.txt', 'w') as f_ptr:
                [f_ptr.write(f"# {m}\n") for m in _msg_arr]
                [f_ptr.write(f"[{dt}] {a:8.4f} {s:.4f}: {k}\n") for dt, a, s, k in _fid_arr]
            # with
        msg_arr = [f"  num_step_arr   : {self.num_step_arr}",
                   f"  grad_method_arr: {self.grad_method_arr}"]
        fid_arr = []
        for num_step in self.num_step_arr:
            self.num_step = num_step
            for grad_method in self.grad_method_arr:
                self.grad_method = grad_method
                key, avg, std = self.sample_times()
                dtstr = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                fid_arr.append([dtstr, avg, std, key])
                save_result(msg_arr, fid_arr)
            # for
        # for

    def alpha_bar_all(self):
        def save_ab_file(file_path):
            ts_list = self.real_seq
            ts_list = th.tensor(ts_list).to(self.device)
            ab_list = self.schedule.ts2ab(ts_list)
            ab_list = ab_list.squeeze(1)
            if len(ab_list) != self.num_step:
                raise Exception(f"alpha_bar count {len(ab_list)} not match steps {self.num_step}")
            with open(file_path, 'w') as f_ptr:
                f_ptr.write(f"# num_step   : {self.num_step}\n")
                f_ptr.write(f"# grad_method: {self.grad_method}\n")
                f_ptr.write(f"\n")
                f_ptr.write(f"# alpha_bar : index\n")
                for ab, ts in zip(ab_list, ts_list):
                    f_ptr.write(f"{ab:.8f}  : {ts:10.5f}\n")
            # with
        # def
        ab_dir = self.args.ab_original_dir or '.'
        if not os.path.exists(ab_dir):
            log_info(f"os.makedirs({ab_dir})")
            os.makedirs(ab_dir)
        for num_step in self.num_step_arr:
            self.num_step = num_step
            for grad_method in self.grad_method_arr:
                self.grad_method = grad_method
                self.sample(total_num=1)
                key = self.config_key_str()
                f_path = os.path.join(ab_dir, f"{key}.txt")
                save_ab_file(f_path)
                log_info(f"File saved: {f_path}")
            # for
        # for

    def config_key_str(self):
        ks = f"s{self.num_step:02d}_{self.grad_method}"
        return ks

    @staticmethod
    def load_predefined_aap(f_path: str, meta_dict=None):
        if not os.path.exists(f_path):
            raise Exception(f"File not found: {f_path}")
        if not os.path.isfile(f_path):
            raise Exception(f"Not file: {f_path}")
        if meta_dict is None:
            meta_dict = {}
        log_info(f"Load file: {f_path}")
        with open(f_path, 'r') as f_ptr:
            lines = f_ptr.readlines()
        cnt_empty = 0
        cnt_comment = 0
        ab_arr = []  # alpha_bar array
        ts_arr = []  # timestep array
        for line in lines:
            line = line.strip()
            if line == '':
                cnt_empty += 1
                continue
            if line.startswith('#'):  # line is like "# order     : 2"
                cnt_comment += 1
                arr = line[1:].strip().split(':')
                key = arr[0].strip()
                if key in meta_dict: meta_dict[key] = arr[1].strip()
                continue
            arr = line.split(':')
            ab, ts = float(arr[0]), float(arr[1])
            ab_arr.append(ab)
            ts_arr.append(ts)
        ab2s = lambda ff: ' '.join([f"{f:8.6f}" for f in ff])
        ts2s = lambda ff: ' '.join([f"{f:10.5f}" for f in ff])
        log_info(f"  cnt_empty  : {cnt_empty}")
        log_info(f"  cnt_comment: {cnt_comment}")
        log_info(f"  cnt_valid  : {len(ab_arr)}")
        log_info(f"  ab[:5]     : [{ab2s(ab_arr[:5])}]")
        log_info(f"  ab[-5:]    : [{ab2s(ab_arr[-5:])}]")
        log_info(f"  ts[:5]     : [{ts2s(ts_arr[:5])}]")
        log_info(f"  ts[-5:]    : [{ts2s(ts_arr[-5:])}]")
        for k, v in meta_dict.items():
            log_info(f"  {k:11s}: {v}")
        return ab_arr, ts_arr

    def sample_times(self, times=None, aap_file=None):
        args = self.args
        times = times or args.repeat_times
        fid_arr = []
        input1, input2 = args.fid_input1 or 'cifar10-train', args.sample_output_dir
        ss = self.config_key_str()
        for i in range(times):
            self.sample(aap_file=aap_file)
            ss = self.config_key_str()  # get ss again as config may change due to aap_file.
            log_info(f"{ss}-{i}/{times} => FID calculating...")
            log_info(f"  input1: {input1}")
            log_info(f"  input2: {input2}")
            metrics_dict = torch_fidelity.calculate_metrics(
                input1=input1,
                input2=input2,
                cuda=True,
                isc=False,
                fid=True,
                kid=False,
                verbose=False,
                samples_find_deep=True,
            )
            fid = metrics_dict['frechet_inception_distance']
            log_info(f"{ss}-{i}/{times} => FID: {fid:.6f}")
            fid_arr.append(fid)
        # for
        avg, std = np.mean(fid_arr), np.std(fid_arr)
        return ss, avg, std

    def sample(self, total_num=None, aap_file=None):
        args = self.args
        image_dir = self.args.sample_output_dir
        if not os.path.exists(image_dir):
            log_info(f"os.makedirs({image_dir})")
            os.makedirs(image_dir)

        model = self.model
        pflow = True if self.grad_method == 'PF' else False

        b_sz = args.sample_batch_size
        total_num = total_num or args.sample_count

        if aap_file:
            meta_dict = {"num_step": None, "grad_method": None}
            ab_arr, _ = self.load_predefined_aap(aap_file, meta_dict)
            self.num_step = int(meta_dict['num_step'])
            self.grad_method = meta_dict['grad_method']
            ab_arr_tensor = th.tensor(ab_arr, device=self.device)
            ts_arr_tensor = self.schedule.ab2ts(ab_arr_tensor)
            ts_arr_tensor = ts_arr_tensor.squeeze(1).cpu().numpy()
            seq_full = np.append([0], ts_arr_tensor)
        else:
            # if num_step is 15, then skip will be 66.
            # and range(0, self.diffusion_step, skip) will have 16 elements
            # So here we do not use builtin range() but to use np.linspace()
            # skip = self.diffusion_step // self.num_step
            tmp = np.linspace(0, 1000, num=self.num_step, endpoint=False)
            seq_full = np.append(tmp, [999])
        seq = seq_full[1:]
        seq_next = seq_full[:-1]
        self.real_seq = seq
        self.schedule = Schedule(self.args, self.config['Schedule'], self.grad_method)
        log_info(f"SampleRunner::sample()...")
        log_info(f"  grad_method   : {self.grad_method}")
        log_info(f"  num_step      : {self.num_step}")
        log_info(f"  diffusion_step: {self.diffusion_step}")
        log_info(f"  seq.len       : {len(seq)}")
        log_info(f"  seq[0]        : {seq[0]:.6f}")
        log_info(f"  seq[-1]       : {seq[-1]:.6f}")
        log_info(f"  seq_next.len  : {len(seq_next)}")
        log_info(f"  seq_next[0]   : {seq_next[0]:.6f}")
        log_info(f"  seq_next[-1]  : {seq_next[-1]:.6f}")
        log_info(f"  new schedule  : {self.grad_method}")

        dc = self.config['Dataset']  # dataset config
        ch, h, w = dc['channels'], dc['image_size'], dc['image_size']
        my_iter = range((total_num - 1) // b_sz + 1)
        b_cnt = len(my_iter)
        log_info(f"  total_num : {total_num}")
        log_info(f"  batch_size: {b_sz}")
        log_info(f"  batch_cnt : {b_cnt}")
        log_info(f"  pflow     : {pflow}")
        for b_idx in my_iter:
            log_info(f"B{b_idx:03d}/{b_cnt}...")
            method._batch_idx = b_idx
            s = b_sz if b_idx+1 < b_cnt else total_num - b_idx * b_sz
            noise = th.randn(s, ch, h, w, device=self.device)
            img = self.sample_image_from_noise(noise, seq, seq_next, model, pflow)
            img = inverse_data_transform(dc, img)
            start_num = b_idx * b_sz
            img_path = None
            for i in range(img.shape[0]):
                image_num = start_num + i
                img_path = os.path.join(image_dir, f"{image_num:05d}.png")
                tvu.save_image(img[i], img_path)
            # for
            log_info(f"  saved {s} images: {img_path}")
        # for

    def sample_image_from_noise(self, noise, seq, seq_next, model, pflow=False):
        with th.no_grad():
            if pflow:
                shape = noise.shape
                device = self.device
                tol = 1e-5 if self.num_step > 1 else self.num_step

                def drift_func(t, x):
                    x = th.from_numpy(x.reshape(shape)).to(device).type(th.float32)
                    drift = self.schedule.denoising(x, None, t, model, pflow=pflow)
                    drift = drift.cpu().numpy().reshape((-1,))
                    return drift

                solution = integrate.solve_ivp(drift_func, (1, 1e-3), noise.cpu().numpy().reshape((-1,)),
                                               rtol=tol, atol=tol, method='RK45')
                img = th.tensor(solution.y[:, -1]).reshape(shape).type(th.float32)

            else:
                imgs = [noise]
                start = True
                n = noise.shape[0]

                for i, j in zip(reversed(seq), reversed(seq_next)):
                    t = (th.ones(n) * i).to(self.device)
                    t_next = (th.ones(n) * j).to(self.device)

                    img_t = imgs[-1].to(self.device)
                    img_next = self.schedule.denoising(img_t, t_next, t, model, start, pflow)
                    start = False

                    imgs.append(img_next.to('cpu'))

                img = imgs[-1]

            return img
        # with
    # def
