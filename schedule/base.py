import math
import os

import numpy as np
import torch
from torch import nn, Tensor

import utils

class ScheduleBase:
    @staticmethod
    def load_floats(f_path, log_fn=utils.log_info):
        if not os.path.exists(f_path):
            raise Exception(f"File not found: {f_path}")
        if not os.path.isfile(f_path):
            raise Exception(f"Not file: {f_path}")
        log_fn(f"load_floats() from file: {f_path}")
        with open(f_path, 'r') as f:
            lines = f.readlines()
        cnt_empty = 0
        cnt_comment = 0
        f_arr = []
        for line in lines:
            line = line.strip()
            if line == '':
                cnt_empty += 1
                continue
            if line.startswith('#'):
                cnt_comment += 1
                continue
            flt = float(line)
            f_arr.append(flt)
        log_fn(f"  cnt_empty  : {cnt_empty}")
        log_fn(f"  cnt_comment: {cnt_comment}")
        log_fn(f"  cnt_valid  : {len(f_arr)}")
        weights = torch.tensor(f_arr, dtype=torch.float64)
        log_fn(f"  weights first 5: {weights[:5].numpy()}")
        log_fn(f"  weights last 5 : {weights[-5:].numpy()}")
        return weights

    @staticmethod
    def accumulate_variance(alpha: Tensor, aacum: Tensor, weight_arr: Tensor, details=False):
        """
        accumulate variance from x_1000 to x_1.
        """
        # delta is to avoid torch error:
        #   RuntimeError: Function 'MulBackward0' returned nan values in its 0th output.
        # Or:
        #   the 2nd epoch will have output: tensor([nan, nan, ,,,])
        # Of that error, a possible reason is: torch tensor 0.sqrt()
        # So here, we make sure alpha > aacum.
        delta = torch.zeros_like(aacum)
        delta[0] = 1e-16
        coef = ((1-aacum).sqrt() - (alpha+delta-aacum).sqrt())**2
        numerator = coef * weight_arr
        sub_var = numerator / aacum
        # sub_var *= alpha
        final_var = torch.sum(sub_var)
        if details:
            return final_var, coef, numerator, sub_var
        return final_var

    @staticmethod
    def get_schedule_from_file(f_path, log_fn=utils.log_info):
        if not os.path.exists(f_path):
            raise Exception(f"File not found: {f_path}")
        if not os.path.isfile(f_path):
            raise Exception(f"Not file: {f_path}")
        log_fn(f"Read file: {f_path}")
        with open(f_path, 'r') as f:
            lines = f.readlines()
        cnt_empty = 0
        cnt_comment = 0
        s_type = ''
        f_arr = []
        for line in lines:
            line = line.strip()
            if line == '':
                cnt_empty += 1
                continue
            if line.startswith('#'):
                cnt_comment += 1
                continue
            if line.startswith('type:'):
                s_type = line.split(':')[1]
                continue
            flt = float(line)
            f_arr.append(flt)
        log_fn(f"  cnt_valid  : {len(f_arr)}")
        log_fn(f"  cnt_empty  : {cnt_empty}")
        log_fn(f"  cnt_comment: {cnt_comment}")
        log_fn(f"  s_type     : {s_type}")
        if s_type == 'alpha':
            alphas = torch.tensor(f_arr).float()
            betas = 1.0 - alphas
            alphas_cumprod = torch.cumprod(alphas, dim=0)
        else:
            raise Exception(f"Unsupported s_type: {s_type} from file {f_path}")
        return betas, alphas, alphas_cumprod

    @staticmethod
    def _get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_timesteps):
        def sigmoid(x):
            return 1 / (np.exp(-x) + 1)

        if beta_schedule == "quad":
            betas = (
                    np.linspace(
                        beta_start ** 0.5,
                        beta_end ** 0.5,
                        num_timesteps,
                        dtype=np.float64,
                    )
                    ** 2
            )
        elif beta_schedule == "linear":
            betas = np.linspace(
                beta_start, beta_end, num_timesteps, dtype=np.float64
            )
        elif beta_schedule == "const":
            betas = beta_end * np.ones(num_timesteps, dtype=np.float64)
        elif beta_schedule == "jsd":  # 1/T, 1/(T-1), 1/(T-2), ..., 1
            betas = 1.0 / np.linspace(
                num_timesteps, 1, num_timesteps, dtype=np.float64
            )
        elif beta_schedule == "sigmoid":
            betas = np.linspace(-6, 6, num_timesteps)
            betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
        else:
            raise NotImplementedError(beta_schedule)
        assert betas.shape == (num_timesteps,)
        return betas

    @staticmethod
    def get_alpha_cumprod(beta_schedule, ts_cnt=1000):
        if beta_schedule == "cosine":
            alphas_cumprod = [] # alpha cumulate array
            for i in range(ts_cnt):
                t = i / ts_cnt
                ac = math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2
                alphas_cumprod.append(ac)
            return torch.Tensor(alphas_cumprod).float()
        elif beta_schedule.startswith('cos:'):
            expo_str = beta_schedule.split(':')[1]  # "cos:2.2"
            expo = float(expo_str)
            alphas_cumprod = []  # alpha cumulate array
            for i in range(ts_cnt):
                t = i / ts_cnt
                ac = math.cos((t + 0.008) / 1.008 * math.pi / 2) ** expo
                alphas_cumprod.append(ac)
            return torch.Tensor(alphas_cumprod).float()
        elif beta_schedule.startswith("noise_rt_expo:"):
            # noise is: 1 - alpha_accumulated
            expo_str = beta_schedule.split(':')[1]  # "noise_rt_expo:2.2"
            expo = float(expo_str)
            n_low, n_high = 0.008, 0.999 # old value
            # n_low, n_high = 0.001, 0.9999  # if "noise_rt_expo:1", got FID 27.792929 on CIFAR-10
            sq_root = np.linspace(n_low, n_high, ts_cnt, dtype=np.float64)
            sq_root = torch.from_numpy(sq_root).float()
            if expo != 1.0:
                sq_root = torch.pow(sq_root, expo)
            sq = torch.mul(sq_root, sq_root)
            return 1 - sq
        elif beta_schedule.startswith('aacum_rt_expo:'):
            expo_str = beta_schedule.split(':')[1]  # "aacum_rt_expo:2.2"
            expo = float(expo_str)
            n_high, n_low = 0.9999, 0.0008 # old value
            # n_high, n_low = 0.9999, 0.001
            # given: 0.9999, 0.001
            #   if "aacum_rt_expo:1",   got FID 22.608681 on CIFAR-10
            #   if "aacum_rt_expo:1.5", got FID 49.226592 on CIFAR-10
            sq_root = np.linspace(n_high, n_low, ts_cnt, dtype=np.float64)
            sq_root = torch.from_numpy(sq_root).float()
            if expo != 1.0:
                sq_root = torch.pow(sq_root, expo)
            return torch.mul(sq_root, sq_root)
        elif beta_schedule.startswith('file:'):
            f_path = beta_schedule.split(':')[1]
            betas, alphas, alphas_cumprod = ScheduleBase.get_schedule_from_file(f_path)
        else:
            betas = ScheduleBase._get_beta_schedule(
                beta_schedule=beta_schedule,
                beta_start=0.0001,
                beta_end=0.02,
                num_timesteps=ts_cnt,
            )
            betas = torch.from_numpy(betas).float()
            alphas = 1.0 - betas
            alphas_cumprod = alphas.cumprod(dim=0)
        return alphas_cumprod


class Schedule1Model(nn.Module):
    def __init__(self, out_channels=1000):
        super().__init__()
        # The two-level linear is better than pure nn.Parameter().
        # Pure nn.Parameter() means such:
        #   self.aa_max = torch.nn.Parameter(torch.ones((1,), dtype=torch.float64), requires_grad=True)
        self.out_channels = out_channels
        self.linear1 = torch.nn.Linear(1000,  2000, dtype=torch.float64)
        self.linear2 = torch.nn.Linear(2000,  2000, dtype=torch.float64)
        self.linear3 = torch.nn.Linear(2000,  out_channels, dtype=torch.float64)

        self.linearMax = torch.nn.Sequential(
            torch.nn.Linear(1, 100, dtype=torch.float64),
            torch.nn.Linear(100, 1, dtype=torch.float64),
        )
        # the seed. we choose value 0.5. And it is better than value 1.0
        ones_k = torch.mul(torch.ones((1000,), dtype=torch.float64), 0.5)
        self.seed_k = torch.nn.Parameter(ones_k, requires_grad=False)
        ones_1 = torch.mul(torch.ones((1,), dtype=torch.float64), 0.5)
        self.seed_1 = torch.nn.Parameter(ones_1, requires_grad=False)

    def gradient_clip(self):
        if self.linear1.weight.grad is not None:
            self.linear1.weight.grad = torch.tanh(self.linear1.weight.grad)
            self.linear1.weight.grad = torch.tanh(self.linear1.weight.grad)
        if self.linear2.weight.grad is not None:
            self.linear2.weight.grad = torch.tanh(self.linear2.weight.grad)
            self.linear2.weight.grad = torch.tanh(self.linear2.weight.grad)
        if self.linear3.weight.grad is not None:
            self.linear3.weight.grad = torch.tanh(self.linear3.weight.grad)
            self.linear3.weight.grad = torch.tanh(self.linear3.weight.grad)

    def forward(self, simple_mode=False):
        output = self.linear1(self.seed_k)
        output = self.linear2(output)
        output = self.linear3(output)
        output = torch.softmax(output, dim=0)
        aa_max = self.linearMax(self.seed_1)
        aa_max = torch.sigmoid(aa_max)
        if simple_mode:
            return output, aa_max

        aacum = torch.cumsum(output, dim=0)
        aacum = torch.flip(aacum, dims=(0,))
        aacum = aacum * aa_max
        aa_prev = torch.cat([torch.ones(1).to(aacum.device), aacum[:-1]], dim=0)
        alpha = torch.div(aacum, aa_prev)

        # alpha = [0.370370, 0.392727, 0.414157, 0.434840, 0.457460,
        #          0.481188, 0.506092, 0.532228, 0.559663, 0.588520,
        #          0.618815, 0.650649, 0.684075, 0.719189, 0.756066,
        #          0.794792, 0.835464, 0.878171, 0.923015, 0.970102, ]
        # aacum = [0.000040, 0.000108, 0.000275, 0.000664, 0.001527,
        #          0.003338, 0.006937, 0.013707, 0.025754, 0.046017,
        #          0.078191, 0.126356, 0.194200, 0.283887, 0.394732,
        #          0.522087, 0.656885, 0.786252, 0.895329, 0.970005, ]
        # alpha, aacum = torch.tensor(alpha), torch.tensor(aacum)
        # alpha, aacum = alpha.to(aa_max.device), aacum.to(aa_max.device)

        return alpha, aacum


class ScheduleAlphaModel(nn.Module):
    """Predict alpha"""
    def __init__(self, out_channels=1000, log_fn=utils.log_info):
        super().__init__()
        self.out_channels = out_channels
        self.linear1 = torch.nn.Linear(1000,  2000, dtype=torch.float64)
        self.linear2 = torch.nn.Linear(2000,  2000, dtype=torch.float64)
        self.linear3 = torch.nn.Linear(2000,  out_channels, dtype=torch.float64)

        # the seed. we choose value 0.5. And it is better than value 1.0
        ones_k = torch.mul(torch.ones((1000,), dtype=torch.float64), 0.5)
        self.seed_k = torch.nn.Parameter(ones_k, requires_grad=False)
        log_fn(f"ScheduleAlphaModel()")
        log_fn(f"  out_channels: {self.out_channels}")

    def gradient_clip(self):
        if self.linear1.weight.grad is not None:
            self.linear1.weight.grad = torch.tanh(self.linear1.weight.grad)
            self.linear1.weight.grad = torch.tanh(self.linear1.weight.grad)
        if self.linear2.weight.grad is not None:
            self.linear2.weight.grad = torch.tanh(self.linear2.weight.grad)
            self.linear2.weight.grad = torch.tanh(self.linear2.weight.grad)
        if self.linear3.weight.grad is not None:
            self.linear3.weight.grad = torch.tanh(self.linear3.weight.grad)
            self.linear3.weight.grad = torch.tanh(self.linear3.weight.grad)

    def forward(self):
        output = self.linear1(self.seed_k)
        output = self.linear2(output)
        output = self.linear3(output)
        alpha = torch.sigmoid(output)
        aacum = torch.cumprod(alpha, dim=0)

        return alpha, aacum


class ScheduleParamAlphaModel(nn.Module):
    """
    Predict alpha, but with predefined alpha base
        Predict alpha, but with predefined alpha base.
    This is for "order-1" DPM solver. The paper detail:
        DPM-Solver: A Fast ODE Solver for Diffusion Probabilistic Model Sampling in Around 10 Steps
        NIPS 2022 Cheng Lu
    """
    def __init__(self, alpha=None, alpha_bar=None, learning_portion=0.01, log_fn=utils.log_info):
        super().__init__()
        if alpha is not None:
            a_base = alpha_bar
            a_base = torch.tensor(a_base)
        elif alpha_bar is not None:
            a_bar = torch.tensor(alpha_bar)
            a_tmp = a_bar[1:] / a_bar[:-1]
            a_base = torch.cat([a_bar[0:1], a_tmp], dim=0)
        else:
            raise ValueError(f"Both alpha and alpha_bar are None")
        a_min = torch.min(a_base)
        a_max = torch.max(a_base)
        assert a_min > 0., f"all alpha must be > 0.: a_min: {a_min}"
        assert a_max < 1., f"all alpha must be < 1.: a_max: {a_max}"
        self.out_channels = len(a_base)
        self.learning_portion = learning_portion
        # make sure learning-portion is small enough. Then new alpha won't exceed range of [0, 1]
        _lp = torch.mul(torch.ones_like(a_base, dtype=torch.float64), learning_portion)
        _lp = torch.minimum(1-a_base, _lp)
        _lp = torch.minimum(a_base, _lp)
        _lp = torch.nn.Parameter(_lp, requires_grad=False)
        self._lp = _lp
        self.log_fn = log_fn
        # hard code the alpha base, which is from DPM-Solver
        # a_base = [0.370370, 0.392727, 0.414157, 0.434840, 0.457460,   # by original TS: 49, 99, 149,,,
        #           0.481188, 0.506092, 0.532228, 0.559663, 0.588520,
        #           0.618815, 0.650649, 0.684075, 0.719189, 0.756066,
        #           0.794792, 0.835464, 0.878171, 0.923015, 0.970102, ]
        # ab.reverse()
        #
        # by geometric with ratio 1.07
        # a_base = [0.991657, 0.978209, 0.961940, 0.942770, 0.920657,
        #           0.895610, 0.867686, 0.828529, 0.797675, 0.750600,
        #           0.704142, 0.654832, 0.597398, 0.537781, 0.477242,
        #           0.417018, 0.353107, 0.292615, 0.236593, 0.177778, ]
        self.alpha_base = torch.nn.Parameter(a_base, requires_grad=False)
        self.linear1 = torch.nn.Linear(1000,  2000, dtype=torch.float64)
        self.linear2 = torch.nn.Linear(2000,  2000, dtype=torch.float64)
        self.linear3 = torch.nn.Linear(2000,  self.out_channels, dtype=torch.float64)

        # the seed. we choose value 0.5. And it is better than value 1.0
        ones_k = torch.mul(torch.ones((1000,), dtype=torch.float64), 0.5)
        self.seed_k = torch.nn.Parameter(ones_k, requires_grad=False)
        f2s = lambda arr: ' '.join([f"{f:.6f}" for f in arr])
        log_fn(f"{type(self).__name__}()")
        log_fn(f"  out_channels     : {self.out_channels}")
        log_fn(f"  learning_portion : {self.learning_portion}")
        log_fn(f"  _lp              : {len(self._lp)}")
        log_fn(f"  _lp[:5]          : [{f2s(self._lp[:5])}]")
        log_fn(f"  _lp[-5:]         : [{f2s(self._lp[-5:])}]")
        log_fn(f"  alpha_base       : {len(self.alpha_base)}")
        log_fn(f"  alpha_base[:5]   : [{f2s(self.alpha_base[:5])}]")
        log_fn(f"  alpha_base[-5:]  : [{f2s(self.alpha_base[-5:])}]")

    def gradient_clip(self):
        if self.linear1.weight.grad is not None:
            self.linear1.weight.grad = torch.tanh(self.linear1.weight.grad)
            self.linear1.weight.grad = torch.tanh(self.linear1.weight.grad)
        if self.linear2.weight.grad is not None:
            self.linear2.weight.grad = torch.tanh(self.linear2.weight.grad)
            self.linear2.weight.grad = torch.tanh(self.linear2.weight.grad)
        if self.linear3.weight.grad is not None:
            self.linear3.weight.grad = torch.tanh(self.linear3.weight.grad)
            self.linear3.weight.grad = torch.tanh(self.linear3.weight.grad)

    def forward(self):
        output = self.linear1(self.seed_k)
        output = self.linear2(output)
        output = self.linear3(output)
        output = torch.tanh(output)
        alpha = torch.add(self.alpha_base, output * self._lp)
        aacum = torch.cumprod(alpha, dim=0)

        return alpha, aacum
