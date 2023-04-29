# Copyright 2022 Luping Liu
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys
import math
import torch as th
import torch.nn as nn
import numpy as np

import runner.method as mtd
from utils import log_info


def get_schedule(config, total_step):
    if config['type'] == "quad":
        betas = (np.linspace(config['beta_start'] ** 0.5, config['beta_end'] ** 0.5, total_step, dtype=np.float64) ** 2)
    elif config['type'] == "linear":
        betas = np.linspace(config['beta_start'], config['beta_end'], total_step, dtype=np.float64)
    elif config['type'] == 'cosine':
        betas = betas_for_alpha_bar(total_step, lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2)
    else:
        betas = None

    betas = th.from_numpy(betas).float()
    alphas = 1.0 - betas
    alphas_cump = alphas.cumprod(dim=0)

    return betas, alphas_cump


def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)


def interpolate_fn(x, xp, yp):
    """
    A piecewise linear function y = f(x), using xp and yp as key points.
    We implement f(x) in a differentiable way (i.e. applicable for autograd).
    The function f(x) is well-defined for all x-axis. (For x beyond the bounds of xp,
     we use the outmost points of xp to define the linear function.)

    Args:
        x: PyTorch tensor with shape [N, C], where N is the batch size,
            C is the number of channels (we use C = 1 for DPM-Solver).
        xp: PyTorch tensor with shape [C, K], where K is the number of keypoints.
        yp: PyTorch tensor with shape [C, K].
    Returns:
        The function values f(x), with shape [N, C].
    """
    N, K = x.shape[0], xp.shape[1]
    all_x = th.cat([x.unsqueeze(2), xp.unsqueeze(0).repeat((N, 1, 1))], dim=2)
    sorted_all_x, x_indices = th.sort(all_x, dim=2)
    x_idx = th.argmin(x_indices, dim=2)
    cand_start_idx = x_idx - 1
    start_idx = th.where(
        th.eq(x_idx, 0),
        th.tensor(1, device=x.device),
        th.where(th.eq(x_idx, K), th.tensor(K - 2, device=x.device), cand_start_idx),
    )
    end_idx = th.where(th.eq(start_idx, cand_start_idx), start_idx + 2, start_idx + 1)
    start_x = th.gather(sorted_all_x, dim=2, index=start_idx.unsqueeze(2)).squeeze(2)
    end_x = th.gather(sorted_all_x, dim=2, index=end_idx.unsqueeze(2)).squeeze(2)
    start_idx2 = th.where(
        th.eq(x_idx, 0),
        th.tensor(0, device=x.device),
        th.where(th.eq(x_idx, K), th.tensor(K - 2, device=x.device), cand_start_idx),
    )
    y_positions_expanded = yp.unsqueeze(0).expand(N, -1, -1)
    start_y = th.gather(y_positions_expanded, dim=2, index=start_idx2.unsqueeze(2)).squeeze(2)
    end_y = th.gather(y_positions_expanded, dim=2, index=(start_idx2 + 1).unsqueeze(2)).squeeze(2)
    cand = start_y + (x - start_x) * (end_y - start_y) / (end_x - start_x)
    return cand


class Schedule(object):
    def __init__(self, args, config, gradient_method):
        self.total_step = config['diffusion_step']
        betas, alphas_cump = get_schedule(config, self.total_step)

        device = th.device(args.device)
        self.betas, self.alphas_cump = betas.to(device), alphas_cump.to(device)
        self.alphas_cump_pre = th.cat([th.ones(1).to(device), self.alphas_cump[:-1]], dim=0)
        timesteps = np.arange(self.total_step)
        self.timesteps = th.from_numpy(timesteps).to(device)

        self.method = mtd.choose_method(gradient_method)  # add pflow
        self.ets = None

    def ab2ts(self, alpha_bar):
        """alpha_bar to timestep"""
        if not hasattr(self, '_ab2ts_flag'):
            setattr(self, '_ab2ts_flag', True)
            log_info(f"schedule::ab2ts() called")
        x_arr, y_arr = self.alphas_cump, self.timesteps
        x_arr, y_arr = th.flip(x_arr, [0]), th.flip(y_arr, [0])
        x_arr, y_arr = x_arr.reshape((1, -1)), y_arr.reshape((1, -1))
        x = alpha_bar.reshape((-1, 1))
        y = interpolate_fn(x, x_arr, y_arr)
        return y

    def ts2ab(self, timestep):
        """timestep to alpha_bar"""
        if not hasattr(self, '_ts2ab_flag'):
            setattr(self, '_ts2ab_flag', True)
            log_info(f"schedule::ts2ab() called")
        x_arr, y_arr = self.timesteps, self.alphas_cump
        x_arr, y_arr = x_arr.reshape((1, -1)), y_arr.reshape((1, -1))
        x = timestep.reshape((-1, 1))
        y = interpolate_fn(x, x_arr, y_arr)
        return y

    def diffusion(self, img, t_end, t_start=0, noise=None):
        if noise is None:
            noise = th.randn_like(img)
        alpha = self.alphas_cump.index_select(0, t_end).view(-1, 1, 1, 1)
        img_n = img * alpha.sqrt() + noise * (1 - alpha).sqrt()

        return img_n, noise

    def denoising(self, img_n, t_end, t_start, model, first_step=False, pflow=False):
        if pflow:
            drift = self.method(img_n, t_start, t_end, model, self.betas, self.total_step)

            return drift
        else:
            if first_step:
                self.ets = []
            # img_next = self.method(img_n, t_start, t_end, model, self.alphas_cump, self.ets)
            img_next = self.method(img_n, t_start, t_end, model, self.ts2ab, self.ets)

            return img_next

