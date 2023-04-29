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
                self.schedule = Schedule(self.args, self.config['Schedule'], self.grad_method)
                key, avg, std = self.sample_fid_times()
                dtstr = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                fid_arr.append([dtstr, avg, std, key])
                save_result(msg_arr, fid_arr)
            # for
        # for

    def config_key_str(self):
        ks = f"s{self.num_step:02d}_{self.grad_method}"
        return ks

    def sample_fid_times(self, times=None):
        args = self.args
        times = times or args.repeat_times
        fid_arr = []
        input1, input2 = args.fid_input1 or 'cifar10-train', args.sample_output_dir
        ss = self.config_key_str()
        for i in range(times):
            self.sample()
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

    def sample(self):
        args = self.args
        image_dir = self.args.sample_output_dir
        if not os.path.exists(image_dir):
            log_info(f"os.makedirs({image_dir})")
            os.makedirs(image_dir)

        model = self.model
        pflow = True if self.grad_method == 'PF' else False

        b_sz = args.sample_batch_size
        total_num = args.sample_count

        skip = self.diffusion_step // self.num_step
        seq = range(0, self.diffusion_step, skip)
        log_info(f"SampleRunner::sample()...")
        log_info(f"  grad_method   : {self.grad_method}")
        log_info(f"  num_step      : {self.num_step}")
        log_info(f"  diffusion_step: {self.diffusion_step}")
        log_info(f"  skip          : {skip}")
        log_info(f"  seq length    : {len(seq)}")
        log_info(f"  seq[0]        : {seq[0]}")
        log_info(f"  seq[-1]       : {seq[-1]}")

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
            img = self.sample_image_from_noise(noise, seq, model, pflow)
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

    def sample_image_from_noise(self, noise, seq, model, pflow=False):
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
                seq_next = [-1] + list(seq[:-1])

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
