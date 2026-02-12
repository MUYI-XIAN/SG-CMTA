# -------------------------------------------------------------------------------------
# § 1. 全局导入
# -------------------------------------------------------------------------------------
import argparse
import csv
import math
import os
import pickle
import random
import time
import warnings
from collections import defaultdict
from math import ceil
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.optim
import torch.optim.lr_scheduler as lr_scheduler
from einops import rearrange, reduce
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sksurv.metrics import concordance_index_censored
from torch import Tensor, einsum
from torch.nn import Module
from torch.nn.init import constant_, xavier_normal_, xavier_uniform_
from torch.nn.modules.linear import NonDynamicallyQuantizableLinear as _LinearWithBias
from torch.nn.parameter import Parameter
from torch.overrides import handle_torch_function, has_torch_function
from torch.optim import Optimizer
from torch.utils.data import (
    DataLoader,
    Dataset,
    RandomSampler,
    Sampler,
    SequentialSampler,
    WeightedRandomSampler,
)
from tqdm import tqdm

# --- 新增的导入 ---
try:
    import openslide
    import torchvision.models as models
    from torchvision import transforms
    from PIL import Image
    import cv2
    import matplotlib.pyplot as plt
except ImportError:
    print("=" * 60)
    print("检测到缺失的依赖库。请使用以下命令安装:")
    print("pip install openslide-python torchvision scikit-learn opencv-python-headless matplotlib")
    print("=" * 60)
    raise

# -------------------------------------------------------------------------------------
# § 2. 通用工具模块 (源自: utils/)
# -------------------------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(
        description="Configurations for Survival Analysis on TCGA Data."
    )
    parser.add_argument(
        "--data_root_dir",
        type=str,
        default="./data/svs/",
        help="Data directory to WSI .svs files",
    )
    parser.add_argument(
        "--feature_dir",
        type=str,
        default="./features/",
        help="Directory to store/load pre-extracted features.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="Random seed for reproducible experiment (default: 1)",
    )
    parser.add_argument(
        "--dataset", type=str, default="tcga_luad", help="Which cancer type to use for training."
    )
    parser.add_argument(
        "--log_data", action="store_true", default=True, help="Log data using tensorboard"
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        dest="evaluate",
        help="Evaluate model on test set",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default="",
        metavar="PATH",
        help="Path to latest checkpoint (default: none)",
    )
    parser.add_argument("--patch_level", type=int, default=2, help="SVS文件读取层级 (0是最高分辨率)")
    parser.add_argument("--patch_size", type=int, default=32, help="从SVS文件提取的patch大小")
    parser.add_argument("--feat_extract_batch_size", type=int, default=256, help="特征提取时patch的处理批次大小")
    parser.add_argument("--model", type=str, default="cmta", help="Type of model (Default: cmta)")
    parser.add_argument(
        "--model_size",
        type=str,
        choices=["small", "large"],
        default="small",
        help="Size of some models (Transformer)",
    )
    parser.add_argument(
        "--modal",
        type=str,
        choices=["omic", "path", "pathomic", "cluster", "coattn"],
        default="coattn",
        help="Specifies which modalities to use / collate function in dataloader.",
    )
    parser.add_argument(
        "--fusion",
        type=str,
        choices=["concat", "bilinear"],
        default="concat",
        help="Modality fuison strategy",
    )
    parser.add_argument("--alpha", type=float, default=0.5, help="hyper-parameter of loss function")

    # --- 创新点控制开关 ---
    # 默认开启创新模块（由单一模块升级为“三模块协同套件”），以便进行消融实验
    parser.add_argument(
        "--use_innovation",
        action="store_true",
        default=True,
        help="是否启用三模块协同创新套件 (默认: True)",
    )
    parser.add_argument(
        "--no_innovation",
        action="store_false",
        dest="use_innovation",
        help="关闭创新模块",
    )

    parser.add_argument(
        "--optimizer",
        type=str,
        choices=["SGD", "Adam", "AdamW", "RAdam", "PlainRAdam", "Lookahead"],
        default="Adam",
    )
    parser.add_argument(
        "--scheduler",
        type=str,
        choices=["None", "exp", "step", "plateau", "cosine"],
        default="cosine",
    )
    parser.add_argument("--num_epoch", type=int, default=20, help="Maximum number of epochs to train (default: 20)")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch Size (Default: 1, due to varying bag sizes)")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate (default: 0.0001)")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="Weight decay")
    parser.add_argument(
        "--loss",
        type=str,
        default="nll_surv_l1",
        choices=[
            "ce_surv",
            "nll_surv",
            "cox_surv",
            "nll_surv_kl",
            "nll_surv_mse",
            "nll_surv_l1",
            "nll_surv_cos",
            "nll_surv_ol",
        ],
        help="slide-level classification loss function (default: nll_surv_l1)",
    )
    parser.add_argument("--weighted_sample", action="store_true", default=True, help="Enable weighted sampling")

    args = parser.parse_args()
    return args


def nll_loss(hazards, S, Y, c, alpha=0.4, eps=1e-7):
    batch_size = len(Y)
    Y = Y.view(batch_size, 1)
    c = c.view(batch_size, 1).float()
    if S is None:
        S = torch.cumprod(1 - hazards, dim=1)

    S_padded = torch.cat([torch.ones_like(c), S], 1)
    uncensored_loss = -(1 - c) * (
        torch.log(torch.gather(S_padded, 1, Y).clamp(min=eps))
        + torch.log(torch.gather(hazards, 1, Y).clamp(min=eps))
    )
    censored_loss = -c * torch.log(torch.gather(S_padded, 1, Y + 1).clamp(min=eps))
    neg_l = censored_loss + uncensored_loss
    loss = (1 - alpha) * neg_l + alpha * uncensored_loss
    loss = loss.mean()
    return loss


def ce_loss(hazards, S, Y, c, alpha=0.4, eps=1e-7):
    batch_size = len(Y)
    Y = Y.view(batch_size, 1)
    c = c.view(batch_size, 1).float()
    if S is None:
        S = torch.cumprod(1 - hazards, dim=1)
    S_padded = torch.cat([torch.ones_like(c), S], 1)
    reg = -(1 - c) * (
        torch.log(torch.gather(S_padded, 1, Y) + eps)
        + torch.log(torch.gather(hazards, 1, Y).clamp(min=eps))
    )
    ce_l = -c * torch.log(torch.gather(S, 1, Y).clamp(min=eps)) - (1 - c) * torch.log(
        1 - torch.gather(S, 1, Y).clamp(min=eps)
    )
    loss = (1 - alpha) * ce_l + alpha * reg
    loss = loss.mean()
    return loss


class CrossEntropySurvLoss(object):
    def __init__(self, alpha=0.15):
        self.alpha = alpha

    def __call__(self, hazards, S, Y, c, alpha=None):
        if alpha is None:
            return ce_loss(hazards, S, Y, c, alpha=self.alpha)
        else:
            return ce_loss(hazards, S, Y, c, alpha=alpha)


class NLLSurvLoss(object):
    def __init__(self, alpha=0.15):
        self.alpha = alpha

    def __call__(self, hazards, S, Y, c, alpha=None):
        if alpha is None:
            return nll_loss(hazards, S, Y, c, alpha=self.alpha)
        else:
            return nll_loss(hazards, S, Y, c, alpha=alpha)


class CoxSurvLoss(object):
    def __call__(hazards, S, c, **kwargs):
        current_batch_len = len(S)
        R_mat = np.zeros([current_batch_len, current_batch_len], dtype=int)
        for i in range(current_batch_len):
            for j in range(current_batch_len):
                R_mat[i, j] = S[j] >= S[i]
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        R_mat = torch.FloatTensor(R_mat).to(device)
        theta = hazards.reshape(-1)
        exp_theta = torch.exp(theta)
        loss_cox = -torch.mean(
            (theta - torch.log(torch.sum(exp_theta * R_mat, dim=1))) * (1 - c)
        )
        return loss_cox


class KLLoss(object):
    def __call__(self, y, y_hat):
        return F.kl_div(y_hat.softmax(dim=-1).log(), y.softmax(dim=-1), reduction="sum")


class CosineLoss(object):
    def __call__(self, y, y_hat):
        return 1 - F.cosine_similarity(y, y_hat, dim=1)


class OrthogonalLoss(nn.Module):
    def __init__(self, gamma=0.5):
        super(OrthogonalLoss, self).__init__()
        self.gamma = gamma

    def forward(self, P, P_hat, G, G_hat):
        pos_pairs = (1 - torch.abs(F.cosine_similarity(P.detach(), P_hat, dim=1))) + (
            1 - torch.abs(F.cosine_similarity(G.detach(), G_hat, dim=1))
        )
        neg_pairs = (
            torch.abs(F.cosine_similarity(P, G, dim=1))
            + torch.abs(F.cosine_similarity(P.detach(), G_hat, dim=1))
            + torch.abs(F.cosine_similarity(G.detach(), P_hat, dim=1))
        )
        loss = pos_pairs + self.gamma * neg_pairs
        return loss


def define_loss(args):
    if args.loss == "ce_surv":
        loss = CrossEntropySurvLoss(alpha=0.0)
    elif args.loss == "nll_surv":
        loss = NLLSurvLoss(alpha=0.0)
    elif args.loss == "cox_surv":
        loss = CoxSurvLoss()
    elif args.loss == "nll_surv_kl":
        loss = [NLLSurvLoss(alpha=0.0), KLLoss()]
    elif args.loss == "nll_surv_mse":
        loss = [NLLSurvLoss(alpha=0.0), nn.MSELoss()]
    elif args.loss == "nll_surv_l1":
        loss = [NLLSurvLoss(alpha=0.0), nn.L1Loss()]
    elif args.loss == "nll_surv_cos":
        loss = [NLLSurvLoss(alpha=0.0), CosineLoss()]
    elif args.loss == "nll_surv_ol":
        loss = [NLLSurvLoss(alpha=0.0), OrthogonalLoss(gamma=0.5)]
    else:
        raise NotImplementedError
    return loss


class RAdam(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        self.buffer = [[None, None, None] for ind in range(10)]
        super(RAdam, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(RAdam, self).__setstate__(state)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data.float()
                if grad.is_sparse:
                    raise RuntimeError("RAdam does not support sparse gradients")
                p_data_fp32 = p.data.float()
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p_data_fp32)
                    state["exp_avg_sq"] = torch.zeros_like(p_data_fp32)
                else:
                    state["exp_avg"] = state["exp_avg"].type_as(p_data_fp32)
                    state["exp_avg_sq"] = state["exp_avg_sq"].type_as(p_data_fp32)
                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                beta1, beta2 = group["betas"]
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                state["step"] += 1
                buffered = self.buffer[int(state["step"] % 10)]
                if state["step"] == buffered[0]:
                    N_sma, step_size = buffered[1], buffered[2]
                else:
                    buffered[0] = state["step"]
                    beta2_t = beta2 ** state["step"]
                    N_sma_max = 2 / (1 - beta2) - 1
                    N_sma = N_sma_max - 2 * state["step"] * beta2_t / (1 - beta2_t)
                    buffered[1] = N_sma
                    if N_sma >= 5:
                        step_size = (
                            group["lr"]
                            * math.sqrt(
                                (1 - beta2_t)
                                * (N_sma - 4)
                                / (N_sma_max - 4)
                                * (N_sma - 2)
                                / N_sma
                                * N_sma_max
                                / (N_sma_max - 2)
                            )
                            / (1 - beta1 ** state["step"])
                        )
                    else:
                        step_size = group["lr"] / (1 - beta1 ** state["step"])
                    buffered[2] = step_size
                if group["weight_decay"] != 0 and group["weight_decay"] is not None:
                    p_data_fp32.add_(-group["weight_decay"] * group["lr"], p_data_fp32)
                if N_sma >= 5:
                    denom = exp_avg_sq.sqrt().add_(group["eps"])
                    p_data_fp32.addcdiv_(-step_size, exp_avg, denom)
                else:
                    p_data_fp32.add_(-step_size, exp_avg)
                p.data.copy_(p_data_fp32)
        return loss


class PlainRAdam(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super(PlainRAdam, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(PlainRAdam, self).__setstate__(state)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data.float()
                if grad.is_sparse:
                    raise RuntimeError("RAdam does not support sparse gradients")
                p_data_fp32 = p.data.float()
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p_data_fp32)
                    state["exp_avg_sq"] = torch.zeros_like(p_data_fp32)
                else:
                    state["exp_avg"] = state["exp_avg"].type_as(p_data_fp32)
                    state["exp_avg_sq"] = state["exp_avg_sq"].type_as(p_data_fp32)
                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                beta1, beta2 = group["betas"]
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                state["step"] += 1
                beta2_t = beta2 ** state["step"]
                N_sma_max = 2 / (1 - beta2) - 1
                N_sma = N_sma_max - 2 * state["step"] * beta2_t / (1 - beta2_t)
                if group["weight_decay"] != 0:
                    p_data_fp32.add_(-group["weight_decay"] * group["lr"], p_data_fp32)
                if N_sma >= 5:
                    step_size = (
                        group["lr"]
                        * math.sqrt(
                            (1 - beta2_t)
                            * (N_sma - 4)
                            / (N_sma_max - 4)
                            * (N_sma - 2)
                            / N_sma
                            * N_sma_max
                            / (N_sma_max - 2)
                        )
                        / (1 - beta1 ** state["step"])
                    )
                    denom = exp_avg_sq.sqrt().add_(group["eps"])
                    p_data_fp32.addcdiv_(-step_size, exp_avg, denom)
                else:
                    step_size = group["lr"] / (1 - beta1 ** state["step"])
                    p_data_fp32.add_(-step_size, exp_avg)
                p.data.copy_(p_data_fp32)
        return loss


class Lookahead(Optimizer):
    def __init__(self, base_optimizer, alpha=0.5, k=6):
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"Invalid slow update rate: {alpha}")
        if not 1 <= k:
            raise ValueError(f"Invalid lookahead steps: {k}")
        defaults = dict(lookahead_alpha=alpha, lookahead_k=k, lookahead_step=0)
        self.base_optimizer = base_optimizer
        self.param_groups = self.base_optimizer.param_groups
        self.defaults = base_optimizer.defaults
        self.defaults.update(defaults)
        self.state = defaultdict(dict)
        for name, default in defaults.items():
            for group in self.param_groups:
                group.setdefault(name, default)

    def update_slow(self, group):
        for fast_p in group["params"]:
            if fast_p.grad is None:
                continue
            param_state = self.state[fast_p]
            if "slow_buffer" not in param_state:
                param_state["slow_buffer"] = torch.empty_like(fast_p.data)
                param_state["slow_buffer"].copy_(fast_p.data)
            slow = param_state["slow_buffer"]
            slow.add_(group["lookahead_alpha"], fast_p.data - slow)
            fast_p.data.copy_(slow)

    def sync_lookahead(self):
        for group in self.param_groups:
            self.update_slow(group)

    def step(self, closure=None):
        loss = self.base_optimizer.step(closure)
        for group in self.param_groups:
            group["lookahead_step"] += 1
            if group["lookahead_step"] % group["lookahead_k"] == 0:
                self.update_slow(group)
        return loss

    def state_dict(self):
        fast_state_dict = self.base_optimizer.state_dict()
        slow_state = {
            (id(k) if isinstance(k, torch.Tensor) else k): v for k, v in self.state.items()
        }
        fast_state = fast_state_dict["state"]
        param_groups = fast_state_dict["param_groups"]
        return {"state": fast_state, "slow_state": slow_state, "param_groups": param_groups}

    def load_state_dict(self, state_dict):
        fast_state_dict = {"state": state_dict["state"], "param_groups": state_dict["param_groups"]}
        self.base_optimizer.load_state_dict(fast_state_dict)
        slow_state_new = False
        if "slow_state" not in state_dict:
            print("Loading state_dict from optimizer without Lookahead applied.")
            state_dict["slow_state"] = defaultdict(dict)
            slow_state_new = True
        slow_state_dict = {"state": state_dict["slow_state"], "param_groups": state_dict["param_groups"]}
        super(Lookahead, self).load_state_dict(slow_state_dict)
        self.param_groups = self.base_optimizer.param_groups
        if slow_state_new:
            for name, default in self.defaults.items():
                for group in self.param_groups:
                    group.setdefault(name, default)


def define_optimizer(args, model):
    if args.optimizer == "SGD":
        optimizer = torch.optim.SGD(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.lr,
            momentum=0.9,
            weight_decay=args.weight_decay,
        )
    elif args.optimizer == "AdamW":
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
    elif args.optimizer == "Adam":
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
    elif args.optimizer == "RAdam":
        optimizer = RAdam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
    elif args.optimizer == "PlainRAdam":
        optimizer = PlainRAdam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
    elif args.optimizer == "Lookahead":
        base_optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
        optimizer = Lookahead(base_optimizer)
    else:
        raise NotImplementedError("Optimizer [{}] is not implemented".format(args.optimizer))
    return optimizer


def define_scheduler(args, optimizer):
    if args.scheduler == "exp":
        scheduler = lr_scheduler.ExponentialLR(optimizer, 0.1, last_epoch=-1)
    elif args.scheduler == "step":
        scheduler = lr_scheduler.StepLR(optimizer, step_size=args.num_epoch / 2, gamma=0.1)
    elif args.scheduler == "plateau":
        scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.2, threshold=0.01, patience=5
        )
    elif args.scheduler == "cosine":
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epoch, eta_min=0)
    elif args.scheduler == "None":
        scheduler = None
    else:
        raise NotImplementedError("Scheduler [{}] is not implemented".format(args.scheduler))
    return scheduler


def collate_surv_vis(batch):
    item = batch[0]
    img = item[0].unsqueeze(0)
    coords = item[1]
    omic_data = [item[i].unsqueeze(0).type(torch.FloatTensor) for i in range(2, 8)]
    label = torch.LongTensor([item[8]])
    event_time = np.array([item[9]])
    c = torch.FloatTensor([item[10]])
    return [img, coords] + omic_data + [label, event_time, c]


def make_weights_for_balanced_classes_split(dataset):
    N = float(len(dataset))
    weight_per_class = [
        (N / len(dataset.slide_cls_ids[c]) if len(dataset.slide_cls_ids[c]) > 0 else 0)
        for c in range(len(dataset.slide_cls_ids))
    ]
    weight = [0] * int(N)
    for idx in range(len(dataset)):
        y = dataset.getlabel(idx)
        weight[idx] = weight_per_class[y]
    return torch.DoubleTensor(weight)


def get_split_loader(split_dataset, training=False, weighted=False, batch_size=1):
    kwargs = {"num_workers": 0, "pin_memory": True} if torch.cuda.is_available() else {}
    collate_fn = collate_surv_vis
    if training:
        if weighted:
            weights = make_weights_for_balanced_classes_split(split_dataset)
            loader = DataLoader(
                split_dataset,
                batch_size=batch_size,
                sampler=WeightedRandomSampler(weights, len(weights)),
                collate_fn=collate_fn,
                **kwargs,
            )
        else:
            loader = DataLoader(
                split_dataset,
                batch_size=batch_size,
                sampler=RandomSampler(split_dataset),
                collate_fn=collate_fn,
                **kwargs,
            )
    else:
        loader = DataLoader(
            split_dataset,
            batch_size=batch_size,
            sampler=SequentialSampler(split_dataset),
            collate_fn=collate_fn,
            **kwargs,
        )
    return loader


def set_seed(seed=7):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


# -------------------------------------------------------------------------------------
# § 3. 模型组件
# -------------------------------------------------------------------------------------
def initialize_weights(module):
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()
        if isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


class BilinearFusion(nn.Module):
    def __init__(
        self,
        skip=0,
        use_bilinear=0,
        gate1=1,
        gate2=1,
        dim1=128,
        dim2=128,
        scale_dim1=1,
        scale_dim2=1,
        mmhid=256,
        dropout_rate=0.25,
    ):
        super(BilinearFusion, self).__init__()
        self.skip = skip
        self.use_bilinear = use_bilinear
        self.gate1 = gate1
        self.gate2 = gate2
        dim1_og, dim2_og, dim1, dim2 = dim1, dim2, dim1 // scale_dim1, dim2 // scale_dim2
        skip_dim = dim1_og + dim2_og if skip else 0
        self.linear_h1 = nn.Sequential(nn.Linear(dim1_og, dim1), nn.ReLU())
        self.linear_z1 = (
            nn.Bilinear(dim1_og, dim2_og, dim1)
            if use_bilinear
            else nn.Sequential(nn.Linear(dim1_og + dim2_og, dim1))
        )
        self.linear_o1 = nn.Sequential(nn.Linear(dim1, dim1), nn.ReLU(), nn.Dropout(p=dropout_rate))
        self.linear_h2 = nn.Sequential(nn.Linear(dim2_og, dim2), nn.ReLU())
        self.linear_z2 = (
            nn.Bilinear(dim1_og, dim2_og, dim2)
            if use_bilinear
            else nn.Sequential(nn.Linear(dim1_og + dim2_og, dim2))
        )
        self.linear_o2 = nn.Sequential(nn.Linear(dim2, dim2), nn.ReLU(), nn.Dropout(p=dropout_rate))
        self.post_fusion_dropout = nn.Dropout(p=dropout_rate)
        self.encoder1 = nn.Sequential(
            nn.Linear((dim1 + 1) * (dim2 + 1), 256), nn.ReLU(), nn.Dropout(p=dropout_rate)
        )
        self.encoder2 = nn.Sequential(
            nn.Linear(256 + skip_dim, mmhid), nn.ReLU(), nn.Dropout(p=dropout_rate)
        )

    def forward(self, vec1, vec2):
        if self.gate1:
            h1 = self.linear_h1(vec1)
            z1 = self.linear_z1(vec1, vec2) if self.use_bilinear else self.linear_z1(
                torch.cat((vec1, vec2), dim=1)
            )
            o1 = self.linear_o1(nn.Sigmoid()(z1) * h1)
        else:
            h1 = self.linear_h1(vec1)
            o1 = self.linear_o1(h1)
        if self.gate2:
            h2 = self.linear_h2(vec2)
            z2 = self.linear_z2(vec1, vec2) if self.use_bilinear else self.linear_z2(
                torch.cat((vec1, vec2), dim=1)
            )
            o2 = self.linear_o2(nn.Sigmoid()(z2) * h2)
        else:
            h2 = self.linear_h2(vec2)
            o2 = self.linear_o2(h2)
        o1 = torch.cat((o1, torch.cuda.FloatTensor(o1.shape[0], 1).fill_(1)), 1)
        o2 = torch.cat((o2, torch.cuda.FloatTensor(o2.shape[0], 1).fill_(1)), 1)
        o12 = torch.bmm(o1.unsqueeze(2), o2.unsqueeze(1)).flatten(start_dim=1)
        out = self.post_fusion_dropout(o12)
        out = self.encoder1(out)
        if self.skip:
            out = torch.cat((out, vec1, vec2), 1)
        out = self.encoder2(out)
        return out


def SNN_Block(dim1, dim2, dropout=0.25):
    return nn.Sequential(nn.Linear(dim1, dim2), nn.ELU(), nn.AlphaDropout(p=dropout, inplace=False))


def init_max_weights(module):
    for m in module.modules():
        if type(m) == nn.Linear:
            stdv = 1.0 / math.sqrt(m.weight.size(1))
            m.weight.data.normal_(0, stdv)
            m.bias.data.zero_()


def exists(val):
    return val is not None


def moore_penrose_iter_pinv(x, iters=6):
    device = x.device
    abs_x = torch.abs(x)
    col = abs_x.sum(dim=-1)
    row = abs_x.sum(dim=-2)
    z = rearrange(x, "... i j -> ... j i") / (torch.max(col) * torch.max(row))
    I = torch.eye(x.shape[-1], device=device)
    I = rearrange(I, "i j -> () i j")
    for _ in range(iters):
        xz = x @ z
        z = 0.25 * z @ (13 * I - (xz @ (15 * I - (xz @ (7 * I - xz)))))
    return z


class NystromAttention(nn.Module):
    def __init__(
        self,
        dim,
        dim_head=64,
        heads=8,
        num_landmarks=256,
        pinv_iterations=6,
        residual=True,
        residual_conv_kernel=33,
        eps=1e-8,
        dropout=0.0,
    ):
        super().__init__()
        self.eps = eps
        inner_dim = heads * dim_head
        self.num_landmarks = num_landmarks
        self.pinv_iterations = pinv_iterations
        self.heads = heads
        self.scale = dim_head**-0.5
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
        self.residual = residual
        if residual:
            kernel_size = residual_conv_kernel
            padding = residual_conv_kernel // 2
            self.res_conv = nn.Conv2d(
                heads, heads, (kernel_size, 1), padding=(padding, 0), groups=heads, bias=False
            )

    def forward(self, x, mask=None, return_attn=False):
        b, n, _, h, m, iters, eps = (
            *x.shape,
            self.heads,
            self.num_landmarks,
            self.pinv_iterations,
            self.eps,
        )
        remainder = n % m
        if remainder > 0:
            padding = m - (n % m)
            x = F.pad(x, (0, 0, padding, 0), value=0)
            if exists(mask):
                mask = F.pad(mask, (padding, 0), value=False)
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=h), (q, k, v))
        if exists(mask):
            mask = rearrange(mask, "b n -> b () n")
            q, k, v = map(lambda t: t * mask[..., None], (q, k, v))
        q = q * self.scale
        l = ceil(n / m)
        landmark_einops_eq = "... (n l) d -> ... n d"
        q_landmarks = reduce(q, landmark_einops_eq, "sum", l=l)
        k_landmarks = reduce(k, landmark_einops_eq, "sum", l=l)
        divisor = l
        if exists(mask):
            mask_landmarks_sum = reduce(mask, "... (n l) -> ... n", "sum", l=l)
            divisor = mask_landmarks_sum[..., None] + eps
            mask_landmarks = mask_landmarks_sum > 0
        q_landmarks /= divisor
        k_landmarks /= divisor
        einops_eq = "... i d, ... j d -> ... i j"
        sim1 = einsum(einops_eq, q, k_landmarks)
        sim2 = einsum(einops_eq, q_landmarks, k_landmarks)
        sim3 = einsum(einops_eq, q_landmarks, k)
        if exists(mask):
            mask_value = -torch.finfo(q.dtype).max
            sim1.masked_fill_(~(mask[..., None] * mask_landmarks[..., None, :]), mask_value)
            sim2.masked_fill_(~(mask_landmarks[..., None] * mask_landmarks[..., None, :]), mask_value)
            sim3.masked_fill_(~(mask_landmarks[..., None] * mask[..., None, :]), mask_value)
        attn1, attn2, attn3 = map(lambda t: t.softmax(dim=-1), (sim1, sim2, sim3))
        attn2_inv = moore_penrose_iter_pinv(attn2, iters)
        out = (attn1 @ attn2_inv) @ (attn3 @ v)
        if self.residual:
            out += self.res_conv(v)
        out = rearrange(out, "b h n d -> b n (h d)", h=h)
        out = self.to_out(out)
        out = out[:, -n:]
        if return_attn:
            attn = attn1 @ attn2_inv @ attn3
            return out, attn
        return out


def multi_head_attention_forward(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    embed_dim_to_check: int,
    num_heads: int,
    in_proj_weight: Tensor,
    in_proj_bias: Tensor,
    bias_k: Optional[Tensor],
    bias_v: Optional[Tensor],
    add_zero_attn: bool,
    dropout_p: float,
    out_proj_weight: Tensor,
    out_proj_bias: Tensor,
    training: bool = True,
    key_padding_mask: Optional[Tensor] = None,
    need_weights: bool = True,
    need_raw: bool = True,
    attn_mask: Optional[Tensor] = None,
    use_separate_proj_weight: bool = False,
    q_proj_weight: Optional[Tensor] = None,
    k_proj_weight: Optional[Tensor] = None,
    v_proj_weight: Optional[Tensor] = None,
    static_k: Optional[Tensor] = None,
    static_v: Optional[Tensor] = None,
):
    tens_ops = (
        query,
        key,
        value,
        in_proj_weight,
        in_proj_bias,
        bias_k,
        bias_v,
        out_proj_weight,
        out_proj_bias,
    )
    if has_torch_function(tens_ops):
        return handle_torch_function(
            multi_head_attention_forward,
            tens_ops,
            query,
            key,
            value,
            embed_dim_to_check,
            num_heads,
            in_proj_weight,
            in_proj_bias,
            bias_k,
            bias_v,
            add_zero_attn,
            dropout_p,
            out_proj_weight,
            out_proj_bias,
            training=training,
            key_padding_mask=key_padding_mask,
            need_weights=need_weights,
            need_raw=need_raw,
            attn_mask=attn_mask,
            use_separate_proj_weight=use_separate_proj_weight,
            q_proj_weight=q_proj_weight,
            k_proj_weight=k_proj_weight,
            v_proj_weight=v_proj_weight,
            static_k=static_k,
            static_v=static_v,
        )
    tgt_len, bsz, embed_dim = query.size()
    assert embed_dim == embed_dim_to_check
    assert key.size(0) == value.size(0) and key.size(1) == value.size(1)
    head_dim = embed_dim // num_heads
    assert head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
    scaling = float(head_dim) ** -0.5
    if not use_separate_proj_weight:
        if (query is key or torch.equal(query, key)) and (key is value or torch.equal(key, value)):
            q, k, v = F.linear(query, in_proj_weight, in_proj_bias).chunk(3, dim=-1)
        elif key is value or torch.equal(key, value):
            _b = in_proj_bias
            _start = 0
            _end = embed_dim
            _w = in_proj_weight[_start:_end, :]
            if _b is not None:
                _b = _b[_start:_end]
            q = F.linear(query, _w, _b)
            if key is None:
                assert value is None
                k = None
                v = None
            else:
                _b = in_proj_bias
                _start = embed_dim
                _end = None
                _w = in_proj_weight[_start:, :]
                if _b is not None:
                    _b = _b[_start:]
                k, v = F.linear(key, _w, _b).chunk(2, dim=-1)
        else:
            _b = in_proj_bias
            _start = 0
            _end = embed_dim
            _w = in_proj_weight[_start:_end, :]
            if _b is not None:
                _b = _b[_start:_end]
            q = F.linear(query, _w, _b)
            _b = in_proj_bias
            _start = embed_dim
            _end = embed_dim * 2
            _w = in_proj_weight[_start:_end, :]
            if _b is not None:
                _b = _b[_start:_end]
            k = F.linear(key, _w, _b)
            _b = in_proj_bias
            _start = embed_dim * 2
            _end = None
            _w = in_proj_weight[_start:, :]
            if _b is not None:
                _b = _b[_start:]
            v = F.linear(value, _w, _b)
    else:
        q_proj_weight_non_opt = torch.jit._unwrap_optional(q_proj_weight)
        len1, len2 = q_proj_weight_non_opt.size()
        assert len1 == embed_dim and len2 == query.size(-1)
        k_proj_weight_non_opt = torch.jit._unwrap_optional(k_proj_weight)
        len1, len2 = k_proj_weight_non_opt.size()
        assert len1 == embed_dim and len2 == key.size(-1)
        v_proj_weight_non_opt = torch.jit._unwrap_optional(v_proj_weight)
        len1, len2 = v_proj_weight_non_opt.size()
        assert len1 == embed_dim and len2 == value.size(-1)
        if in_proj_bias is not None:
            q = F.linear(query, q_proj_weight_non_opt, in_proj_bias[0:embed_dim])
            k = F.linear(
                key,
                k_proj_weight_non_opt,
                in_proj_bias[embed_dim : (embed_dim * 2)],
            )
            v = F.linear(
                value,
                v_proj_weight_non_opt,
                in_proj_bias[(embed_dim * 2) :],
            )
        else:
            q = F.linear(query, q_proj_weight_non_opt, in_proj_bias)
            k = F.linear(key, k_proj_weight_non_opt, in_proj_bias)
            v = F.linear(value, v_proj_weight_non_opt, in_proj_bias)
    q = q * scaling
    if attn_mask is not None:
        assert (
            attn_mask.dtype == torch.float32
            or attn_mask.dtype == torch.float64
            or attn_mask.dtype == torch.float16
            or attn_mask.dtype == torch.uint8
            or attn_mask.dtype == torch.bool
        ), "Only float, byte, and bool types are supported for attn_mask, not {}".format(
            attn_mask.dtype
        )
        if attn_mask.dtype == torch.uint8:
            warnings.warn(
                "Byte tensor for attn_mask in nn.MultiheadAttention is deprecated. Use bool tensor instead."
            )
            attn_mask = attn_mask.to(torch.bool)
        if attn_mask.dim() == 2:
            attn_mask = attn_mask.unsqueeze(0)
            if list(attn_mask.size()) != [1, query.size(0), key.size(0)]:
                raise RuntimeError("The size of the 2D attn_mask is not correct.")
        elif attn_mask.dim() == 3:
            if list(attn_mask.size()) != [bsz * num_heads, query.size(0), key.size(0)]:
                raise RuntimeError("The size of the 3D attn_mask is not correct.")
        else:
            raise RuntimeError("attn_mask's dimension {} is not supported".format(attn_mask.dim()))
    if key_padding_mask is not None and key_padding_mask.dtype == torch.uint8:
        warnings.warn(
            "Byte tensor for key_padding_mask in nn.MultiheadAttention is deprecated. Use bool tensor instead."
        )
        key_padding_mask = key_padding_mask.to(torch.bool)
    if bias_k is not None and bias_v is not None:
        if static_k is None and static_v is None:
            k = torch.cat([k, bias_k.repeat(1, bsz, 1)])
            v = torch.cat([v, bias_v.repeat(1, bsz, 1)])
            if attn_mask is not None:
                attn_mask = F.pad(attn_mask, (0, 1))
            if key_padding_mask is not None:
                key_padding_mask = F.pad(key_padding_mask, (0, 1))
        else:
            assert static_k is None, "bias cannot be added to static key."
            assert static_v is None, "bias cannot be added to static value."
    else:
        assert bias_k is None
        assert bias_v is None
    q = q.contiguous().view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)
    if k is not None:
        k = k.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
    if v is not None:
        v = v.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
    if static_k is not None:
        assert static_k.size(0) == bsz * num_heads
        assert static_k.size(2) == head_dim
        k = static_k
    if static_v is not None:
        assert static_v.size(0) == bsz * num_heads
        assert static_v.size(2) == head_dim
        v = static_v
    src_len = k.size(1)
    if key_padding_mask is not None:
        assert key_padding_mask.size(0) == bsz
        assert key_padding_mask.size(1) == src_len
    if add_zero_attn:
        src_len += 1
        k = torch.cat(
            [k, torch.zeros((k.size(0), 1) + k.size()[2:], dtype=k.dtype, device=k.device)],
            dim=1,
        )
        v = torch.cat(
            [v, torch.zeros((v.size(0), 1) + v.size()[2:], dtype=v.dtype, device=v.device)],
            dim=1,
        )
        if attn_mask is not None:
            attn_mask = F.pad(attn_mask, (0, 1))
        if key_padding_mask is not None:
            key_padding_mask = F.pad(key_padding_mask, (0, 1))
    attn_output_weights = torch.bmm(q, k.transpose(1, 2))
    assert list(attn_output_weights.size()) == [bsz * num_heads, tgt_len, src_len]
    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_output_weights.masked_fill_(attn_mask, float("-inf"))
        else:
            attn_output_weights += attn_mask
    if key_padding_mask is not None:
        attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
        attn_output_weights = attn_output_weights.masked_fill(
            key_padding_mask.unsqueeze(1).unsqueeze(2), float("-inf")
        )
        attn_output_weights = attn_output_weights.view(bsz * num_heads, tgt_len, src_len)
    attn_output_weights_raw = attn_output_weights
    attn_output_weights = F.softmax(attn_output_weights, dim=-1)
    attn_output_weights = F.dropout(attn_output_weights, p=dropout_p, training=training)
    attn_output = torch.bmm(attn_output_weights, v)
    assert list(attn_output.size()) == [bsz * num_heads, tgt_len, head_dim]
    attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
    attn_output = F.linear(attn_output, out_proj_weight, out_proj_bias)
    if need_weights:
        if need_raw:
            attn_output_weights_raw = attn_output_weights_raw.view(bsz, num_heads, tgt_len, src_len)
            return attn_output, attn_output_weights_raw
        else:
            attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
            return attn_output, attn_output_weights.sum(dim=1) / num_heads
    else:
        return attn_output, None


class MultiheadAttention(Module):
    bias_k: Optional[torch.Tensor]
    bias_v: Optional[torch.Tensor]

    def __init__(
        self,
        embed_dim,
        num_heads,
        dropout=0.0,
        bias=True,
        add_bias_kv=False,
        add_zero_attn=False,
        kdim=None,
        vdim=None,
    ):
        super(MultiheadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self._qkv_same_embed_dim = self.kdim == embed_dim and self.vdim == embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"
        if self._qkv_same_embed_dim is False:
            self.q_proj_weight = Parameter(torch.Tensor(embed_dim, embed_dim))
            self.k_proj_weight = Parameter(torch.Tensor(embed_dim, self.kdim))
            self.v_proj_weight = Parameter(torch.Tensor(embed_dim, self.vdim))
            self.register_parameter("in_proj_weight", None)
        else:
            self.in_proj_weight = Parameter(torch.empty(3 * embed_dim, embed_dim))
            self.register_parameter("q_proj_weight", None)
            self.register_parameter("k_proj_weight", None)
            self.register_parameter("v_proj_weight", None)
        if bias:
            self.in_proj_bias = Parameter(torch.empty(3 * embed_dim))
        else:
            self.register_parameter("in_proj_bias", None)
        self.out_proj = _LinearWithBias(embed_dim, embed_dim)
        if add_bias_kv:
            self.bias_k = Parameter(torch.empty(1, 1, embed_dim))
            self.bias_v = Parameter(torch.empty(1, 1, embed_dim))
        else:
            self.bias_k = self.bias_v = None
        self.add_zero_attn = add_zero_attn
        self._reset_parameters()

    def _reset_parameters(self):
        if self._qkv_same_embed_dim:
            xavier_uniform_(self.in_proj_weight)
        else:
            xavier_uniform_(self.q_proj_weight)
            xavier_uniform_(self.k_proj_weight)
            xavier_uniform_(self.v_proj_weight)
        if self.in_proj_bias is not None:
            constant_(self.in_proj_bias, 0.0)
            constant_(self.out_proj.bias, 0.0)
        if self.bias_k is not None:
            xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            xavier_normal_(self.bias_v)

    def __setstate__(self, state):
        if "_qkv_same_embed_dim" not in state:
            state["_qkv_same_embed_dim"] = True
        super(MultiheadAttention, self).__setstate__(state)

    def forward(self, query, key, value, key_padding_mask=None, need_weights=True, need_raw=True, attn_mask=None):
        if not self._qkv_same_embed_dim:
            return multi_head_attention_forward(
                query,
                key,
                value,
                self.embed_dim,
                self.num_heads,
                self.in_proj_weight,
                self.in_proj_bias,
                self.bias_k,
                self.bias_v,
                self.add_zero_attn,
                self.dropout,
                self.out_proj.weight,
                self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask,
                need_weights=need_weights,
                need_raw=need_raw,
                attn_mask=attn_mask,
                use_separate_proj_weight=True,
                q_proj_weight=self.q_proj_weight,
                k_proj_weight=self.k_proj_weight,
                v_proj_weight=self.v_proj_weight,
            )
        else:
            return multi_head_attention_forward(
                query,
                key,
                value,
                self.embed_dim,
                self.num_heads,
                self.in_proj_weight,
                self.in_proj_bias,
                self.bias_k,
                self.bias_v,
                self.add_zero_attn,
                self.dropout,
                self.out_proj.weight,
                self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask,
                need_weights=need_weights,
                need_raw=need_raw,
                attn_mask=attn_mask,
            )


# -------------------------------------------------------------------------------------
# § 4. 核心模型架构
# -------------------------------------------------------------------------------------

# --- 创新点拆分：三模块协同注意力增强套件（流程与原 DSAM-AF 完全一致） ---
# 原逻辑：feat -> (自生成 pred) -> 前景/背景双流 ASPP -> 拼接 -> Pred_Layer 输出回 in_c -> 返回 enhanced_feat
#
# 现升级为三个“创新点/模块”：
# 1) SGAP：Self-Generated Attention Prior（自生成注意力先验）
# 2) DSEP：Dual-Stream Atrous Context Pyramid（双流空洞多尺度上下文金字塔）
# 3) ARFH：Adaptive Residual Fusion Head（自适应残差融合头）
#
# 注意：仅拆分与命名升级，计算流程、张量变换、残差接口完全保持不变。


class AdaptiveResidualProjectionHead(nn.Module):
    """
    原 Pred_Layer 的严谨改名版本：保持实现不变
    - 输入: concat_dim (前景/背景双流拼接后的通道数)
    - 输出: out_c (= in_c)，以保证与 Transformer 残差连接维度一致
    """
    def __init__(self, in_c=256, out_c=None):
        super(AdaptiveResidualProjectionHead, self).__init__()
        self.output_dim = out_c if out_c is not None else 256
        self.enlayer = nn.Sequential(
            nn.Conv2d(in_c, self.output_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.output_dim),
            nn.ReLU(inplace=True),
        )
        self.outlayer = nn.Sequential(
            nn.Conv2d(self.output_dim, 1, kernel_size=1, stride=1, padding=0),
        )

    def forward(self, x):
        x = self.enlayer(x)
        x1 = self.outlayer(x)
        return x, x1


class AtrousSpatialContextPyramid(nn.Module):
    """
    原 ASPP 的严谨改名版本：保持实现不变
    """
    def __init__(self, in_c, mid_c=None):
        super(AtrousSpatialContextPyramid, self).__init__()
        if mid_c is None:
            mid_c = 256

        self.aspp1 = nn.Sequential(
            nn.Conv2d(in_c, mid_c, 1, 1),
            nn.BatchNorm2d(mid_c),
            nn.ReLU(inplace=True),
        )
        self.aspp2 = nn.Sequential(
            nn.Conv2d(in_c, mid_c, 3, 1, padding=3, dilation=3),
            nn.BatchNorm2d(mid_c),
            nn.ReLU(inplace=True),
        )
        self.aspp3 = nn.Sequential(
            nn.Conv2d(in_c, mid_c, 3, 1, padding=5, dilation=5),
            nn.BatchNorm2d(mid_c),
            nn.ReLU(inplace=True),
        )
        self.aspp4 = nn.Sequential(
            nn.Conv2d(in_c, mid_c, 3, 1, padding=7, dilation=7),
            nn.BatchNorm2d(mid_c),
            nn.ReLU(inplace=True),
        )
        self.output_dim = mid_c * 4

    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x = torch.cat((x1, x2, x3, x4), dim=1)
        return x


class SelfGeneratedAttentionPrior(nn.Module):
    """
    创新点 1 / SGAP:
    Self-Generated Attention Prior（自生成注意力先验）
    - 由特征自身生成 1-channel prior（与原 initial_pred + interpolate + sigmoid 保持一致）
    """
    def __init__(self, in_c: int):
        super(SelfGeneratedAttentionPrior, self).__init__()
        self.initial_pred = nn.Sequential(
            nn.Conv2d(in_c, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        pred = self.initial_pred(feat)

        [_, _, H, W] = feat.size()
        pred = torch.sigmoid(
            F.interpolate(
                pred,
                size=(H, W),
                mode="bilinear",
                align_corners=True,
            )
        )
        return pred


class DualStreamAtrousContextPyramid(nn.Module):
    """
    创新点 2 / DSEP:
    Dual-Stream Atrous Context Pyramid（双流空洞多尺度上下文金字塔）
    - 前景流: feat * pred
    - 背景流: feat * (1 - pred)
    - 每一路均用同构多尺度空洞卷积金字塔（与原两路 ASPP 一致）
    """
    def __init__(self, in_c: int, mid_c: Optional[int] = None):
        super(DualStreamAtrousContextPyramid, self).__init__()
        if mid_c is None:
            mid_c = max(64, in_c // 4)

        self.ff_conv = AtrousSpatialContextPyramid(in_c, mid_c)
        self.bf_conv = AtrousSpatialContextPyramid(in_c, mid_c)

        self.aspp_out_dim = mid_c * 4

    def forward(self, feat: torch.Tensor, pred: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        ff_feat = self.ff_conv(feat * pred)
        bf_feat = self.bf_conv(feat * (1 - pred))
        return ff_feat, bf_feat


class AdaptiveResidualFusionHead(nn.Module):
    """
    创新点 3 / ARFH:
    Adaptive Residual Fusion Head（自适应残差融合头）
    - 将双流金字塔输出拼接后映射回 in_c，以满足残差接口
    - 与原 rgbd_pred_layer（Pred_Layer）逻辑一致
    """
    def __init__(self, in_c: int, aspp_out_dim: int):
        super(AdaptiveResidualFusionHead, self).__init__()
        concat_dim = aspp_out_dim * 2
        self.fusion_head = AdaptiveResidualProjectionHead(in_c=concat_dim, out_c=in_c)

    def forward(self, ff_feat: torch.Tensor, bf_feat: torch.Tensor) -> torch.Tensor:
        enhanced_feat, _ = self.fusion_head(torch.cat((ff_feat, bf_feat), 1))
        return enhanced_feat


# -----------------------------------


class TransLayer(nn.Module):
    def __init__(self, norm_layer=nn.LayerNorm, dim=512):
        super().__init__()
        self.norm = norm_layer(dim)
        self.attn = NystromAttention(
            dim=dim,
            dim_head=dim // 8,
            heads=8,
            num_landmarks=dim // 2,
            pinv_iterations=6,
            residual=True,
            dropout=0.1,
        )

    def forward(self, x, return_attn=False):
        if return_attn:
            attn_output, attn_weights = self.attn(self.norm(x), return_attn=True)
            x = x + attn_output
            return x, attn_weights
        else:
            x = x + self.attn(self.norm(x))
            return x


class PPEG(nn.Module):
    def __init__(self, dim=512, use_innovation=True):
        super(PPEG, self).__init__()
        self.use_innovation = use_innovation

        if self.use_innovation:
            # 三模块协同创新套件（替代原 DSAM-AF，流程保持一致）
            self.sgap = SelfGeneratedAttentionPrior(in_c=dim)
            self.dsep = DualStreamAtrousContextPyramid(in_c=dim, mid_c=max(64, dim // 4))
            self.arfh = AdaptiveResidualFusionHead(in_c=dim, aspp_out_dim=self.dsep.aspp_out_dim)
        else:
            # 标准多尺度卷积位置编码（原逻辑保持）
            self.proj = nn.Conv2d(dim, dim, 7, 1, 7 // 2, groups=dim)
            self.proj1 = nn.Conv2d(dim, dim, 5, 1, 5 // 2, groups=dim)
            self.proj2 = nn.Conv2d(dim, dim, 3, 1, 3 // 2, groups=dim)

    def forward(self, x, H, W):
        B, _, C = x.shape
        cls_token, feat_token = x[:, 0], x[:, 1:]
        cnn_feat = feat_token.transpose(1, 2).view(B, C, H, W)

        feat = cnn_feat

        if self.use_innovation:
            # 1) SGAP：自生成注意力先验
            pred = self.sgap(feat)
            # 2) DSEP：双流多尺度上下文提取
            ff_feat, bf_feat = self.dsep(feat, pred)
            # 3) ARFH：自适应残差融合回归
            pos_encoding_component = self.arfh(ff_feat, bf_feat)
        else:
            pos_encoding_component = self.proj(feat) + self.proj1(feat) + self.proj2(feat)

        x = feat + pos_encoding_component

        x = x.flatten(2).transpose(1, 2)
        x = torch.cat((cls_token.unsqueeze(1), x), dim=1)
        return x


class Transformer_P(nn.Module):
    def __init__(self, feature_dim=512, use_innovation=True):
        super(Transformer_P, self).__init__()
        self.pos_layer = PPEG(dim=feature_dim, use_innovation=use_innovation)
        self.cls_token = nn.Parameter(torch.randn(1, 1, feature_dim))
        nn.init.normal_(self.cls_token, std=1e-6)
        self.layer1 = TransLayer(dim=feature_dim)
        self.layer2 = TransLayer(dim=feature_dim)
        self.norm = nn.LayerNorm(feature_dim)

    def forward(self, features, return_attn=False):
        H = features.shape[1]
        _H, _W = int(np.ceil(np.sqrt(H))), int(np.ceil(np.sqrt(H)))
        add_length = _H * _W - H
        if add_length > 0:
            h = torch.cat([features, features[:, :add_length, :]], dim=1)
        else:
            h = features

        B = h.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1).cuda()
        h = torch.cat((cls_tokens, h), dim=1)

        h = self.layer1(h)
        h = self.pos_layer(h, _H, _W)

        if return_attn:
            h, attn = self.layer2(h, return_attn=True)
            h = self.norm(h)
            return h[:, 0], h[:, 1:], attn
        else:
            h = self.layer2(h)
            h = self.norm(h)
            return h[:, 0], h[:, 1:]


class Transformer_G(nn.Module):
    def __init__(self, feature_dim=512):
        super(Transformer_G, self).__init__()
        self.cls_token = nn.Parameter(torch.randn(1, 1, feature_dim))
        nn.init.normal_(self.cls_token, std=1e-6)
        self.layer1 = TransLayer(dim=feature_dim)
        self.layer2 = TransLayer(dim=feature_dim)
        self.norm = nn.LayerNorm(feature_dim)

    def forward(self, features):
        cls_tokens = self.cls_token.expand(features.shape[0], -1, -1).cuda()
        h = torch.cat((cls_tokens, features), dim=1)
        h = self.layer1(h)
        h = self.layer2(h)
        h = self.norm(h)
        return h[:, 0], h[:, 1:]


class CMTA(nn.Module):
    def __init__(
        self,
        omic_sizes=[100, 200, 300, 400, 500, 600],
        n_classes=4,
        fusion="concat",
        model_size="small",
        use_innovation=True,
    ):
        super(CMTA, self).__init__()
        self.omic_sizes = omic_sizes
        self.n_classes = n_classes
        self.fusion = fusion
        self.size_dict = {
            "pathomics": {"small": [2048, 256, 256], "large": [2048, 512, 256]},
            "genomics": {"small": [1024, 256], "large": [1024, 1024, 1024, 256]},
        }
        hidden = self.size_dict["pathomics"][model_size]
        fc = []
        for idx in range(len(hidden) - 1):
            fc.append(nn.Linear(hidden[idx], hidden[idx + 1]))
            fc.append(nn.ReLU())
            fc.append(nn.Dropout(0.25))
        self.pathomics_fc = nn.Sequential(*fc)
        hidden = self.size_dict["genomics"][model_size]
        sig_networks = []
        for input_dim in omic_sizes:
            fc_omic = [SNN_Block(dim1=input_dim, dim2=hidden[0])]
            for i, _ in enumerate(hidden[1:]):
                fc_omic.append(SNN_Block(dim1=hidden[i], dim2=hidden[i + 1], dropout=0.25))
            sig_networks.append(nn.Sequential(*fc_omic))
        self.genomics_fc = nn.ModuleList(sig_networks)

        # 创新开关：传递至 Transformer_P（其内部 PPEG 已升级为三模块套件）
        self.pathomics_encoder = Transformer_P(feature_dim=hidden[-1], use_innovation=use_innovation)
        self.pathomics_decoder = Transformer_P(feature_dim=hidden[-1], use_innovation=use_innovation)

        self.P_in_G_Att = MultiheadAttention(embed_dim=256, num_heads=1)
        self.G_in_P_Att = MultiheadAttention(embed_dim=256, num_heads=1)
        self.genomics_encoder = Transformer_G(feature_dim=hidden[-1])
        self.genomics_decoder = Transformer_G(feature_dim=hidden[-1])
        if self.fusion == "concat":
            self.mm = nn.Sequential(
                *[
                    nn.Linear(hidden[-1] * 2, hidden[-1]),
                    nn.ReLU(),
                    nn.Linear(hidden[-1], hidden[-1]),
                    nn.ReLU(),
                ]
            )
        elif self.fusion == "bilinear":
            self.mm = BilinearFusion(
                dim1=hidden[-1], dim2=hidden[-1], scale_dim1=8, scale_dim2=8, mmhid=hidden[-1]
            )
        else:
            raise NotImplementedError("Fusion [{}] is not implemented".format(self.fusion))
        self.classifier = nn.Linear(hidden[-1], self.n_classes)
        self.apply(initialize_weights)

    def forward(self, return_attn=False, **kwargs):
        x_path = kwargs["x_path"]
        x_omic = [kwargs["x_omic%d" % i] for i in range(1, 7)]
        genomics_features = [self.genomics_fc[idx].forward(sig_feat) for idx, sig_feat in enumerate(x_omic)]
        genomics_features = torch.stack(genomics_features, dim=0).transpose(0, 1)
        pathomics_features = self.pathomics_fc(x_path).unsqueeze(0)

        attn = None
        if return_attn:
            cls_token_pathomics_encoder, patch_token_pathomics_encoder, attn = self.pathomics_encoder(
                pathomics_features, return_attn=True
            )
        else:
            cls_token_pathomics_encoder, patch_token_pathomics_encoder = self.pathomics_encoder(pathomics_features)

        cls_token_genomics_encoder, patch_token_genomics_encoder = self.genomics_encoder(genomics_features)
        pathomics_in_genomics, _ = self.P_in_G_Att(
            patch_token_pathomics_encoder.transpose(1, 0),
            patch_token_genomics_encoder.transpose(1, 0),
            patch_token_genomics_encoder.transpose(1, 0),
        )
        genomics_in_pathomics, _ = self.G_in_P_Att(
            patch_token_genomics_encoder.transpose(1, 0),
            patch_token_pathomics_encoder.transpose(1, 0),
            patch_token_pathomics_encoder.transpose(1, 0),
        )
        cls_token_pathomics_decoder, _ = self.pathomics_decoder(pathomics_in_genomics.transpose(1, 0))
        cls_token_genomics_decoder, _ = self.genomics_decoder(genomics_in_pathomics.transpose(1, 0))

        if self.fusion == "concat":
            fusion = self.mm(
                torch.concat(
                    (
                        (cls_token_pathomics_encoder + cls_token_pathomics_decoder) / 2,
                        (cls_token_genomics_encoder + cls_token_genomics_decoder) / 2,
                    ),
                    dim=1,
                )
            )
        elif self.fusion == "bilinear":
            fusion = self.mm(
                (cls_token_pathomics_encoder + cls_token_pathomics_decoder) / 2,
                (cls_token_genomics_encoder + cls_token_genomics_decoder) / 2,
            )
        else:
            raise NotImplementedError("Fusion [{}] is not implemented".format(self.fusion))

        logits = self.classifier(fusion)
        hazards = torch.sigmoid(logits)
        S = torch.cumprod(1 - hazards, dim=1)

        if return_attn:
            return (
                hazards,
                S,
                cls_token_pathomics_encoder,
                cls_token_pathomics_decoder,
                cls_token_genomics_encoder,
                cls_token_genomics_decoder,
                attn,
            )
        else:
            return (
                hazards,
                S,
                cls_token_pathomics_encoder,
                cls_token_pathomics_decoder,
                cls_token_genomics_encoder,
                cls_token_genomics_decoder,
            )


# -------------------------------------------------------------------------------------
# § 5. 数据集类
# -------------------------------------------------------------------------------------


class ResNet_Feature_Extractor(nn.Module):
    def __init__(self, pretrained_model="resnet50"):
        super(ResNet_Feature_Extractor, self).__init__()
        if pretrained_model == "resnet50":
            resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
            self.feature_dim = 2048
        else:
            raise NotImplementedError("Only 'resnet50' is supported for now.")

        modules = list(resnet.children())[:-1]
        self.extractor = nn.Sequential(*modules)

        for param in self.extractor.parameters():
            param.requires_grad = False

    def forward(self, x):
        features = self.extractor(x)
        features = features.view(x.size(0), -1)
        return features


def get_tissue_mask(slide: openslide.OpenSlide, level: int = -1):
    if level == -1:
        level = slide.level_count - 1
    thumb = slide.read_region((0, 0), level, slide.level_dimensions[level]).convert("L")
    thumb_np = np.array(thumb)
    _, mask = cv2.threshold(thumb_np, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.zeros_like(mask, dtype=bool), 0.0
    contour_areas = [cv2.contourArea(c) for c in contours]
    max_contour = contours[np.argmax(contour_areas)]
    final_mask = np.zeros_like(mask)
    cv2.drawContours(final_mask, [max_contour], -1, 255, thickness=cv2.FILLED)
    downsample_factor = slide.level_downsamples[level]
    return final_mask.astype(bool), downsample_factor


def extract_and_save_features_for_slide(slide_path, feature_extractor, preprocess, args, output_path):
    try:
        slide = openslide.OpenSlide(slide_path)
    except openslide.OpenSlideError:
        print(f"警告: 无法打开文件 {slide_path}。跳过此样本。")
        return False

    patch_size = args.patch_size
    target_level = args.patch_level

    if target_level >= slide.level_count:
        target_level = slide.level_count - 1
        print(f"警告: 请求的层级 {args.patch_level} 超出范围。使用层级 {target_level}。")

    mask_level = slide.level_count - 1
    tissue_mask, mask_downsample = get_tissue_mask(slide, mask_level)

    scale_factor = slide.level_downsamples[target_level] / mask_downsample

    width, height = slide.level_dimensions[target_level]
    all_patches, all_coords = [], []

    for y in range(0, height, patch_size):
        for x in range(0, width, patch_size):
            mask_x = int((x + patch_size / 2) / scale_factor)
            mask_y = int((y + patch_size / 2) / scale_factor)
            if (
                0 <= mask_y < tissue_mask.shape[0]
                and 0 <= mask_x < tissue_mask.shape[1]
                and tissue_mask[mask_y, mask_x]
            ):
                try:
                    patch = slide.read_region(
                        (
                            int(x * slide.level_downsamples[target_level]),
                            int(y * slide.level_downsamples[target_level]),
                        ),
                        target_level,
                        (patch_size, patch_size),
                    ).convert("RGB")
                    all_patches.append(patch)
                    all_coords.append((x, y))
                except Exception as e:
                    print(f"读取patch出错: {e}, 跳过。")
                    continue

    if not all_patches:
        print(f"警告: 在 {slide_path} 的组织区域内未提取到任何 patch。创建空的特征张量。")
        features_tensor = torch.zeros(1, feature_extractor.feature_dim)
        coords_list = []
    else:
        feature_list = []
        device = next(feature_extractor.parameters()).device
        batch_size = args.feat_extract_batch_size
        with torch.no_grad():
            for i in tqdm(
                range(0, len(all_patches), batch_size),
                desc=f"提取特征 {os.path.basename(slide_path)}",
                leave=False,
                ncols=80,
            ):
                batch = all_patches[i : i + batch_size]
                batch_tensors = torch.stack([preprocess(p) for p in batch]).to(device)
                features = feature_extractor(batch_tensors)
                feature_list.append(features.cpu())
        features_tensor = torch.cat(feature_list, dim=0)
        coords_list = all_coords

    slide.close()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save({"features": features_tensor, "coords": coords_list}, output_path)
    return True


class Survival_SVS_Dataset(Dataset):
    def __init__(self, df, feature_dir, args, apply_sig=False, signatures=None):
        self.slide_data = df.reset_index(drop=True)
        self.feature_dir = feature_dir
        self.args = args
        self.label_col = "survival_months"
        self.patient_dict = self._create_patient_dict()

        self.apply_sig = apply_sig
        self.signatures = signatures if self.apply_sig and signatures is not None else None

        self.metadata = [
            "case_id",
            "slide_id",
            "censorship",
            "survival_months",
            "disc_label",
            "label",
            "slide_filename",
        ]
        genomic_cols = [col for col in self.slide_data.columns if col not in self.metadata]
        numeric_features_df = self.slide_data[genomic_cols].select_dtypes(include=np.number)
        self.genomic_features = numeric_features_df

        if self.signatures is not None:
            self.omic_names = []
            for col in self.signatures.columns:
                omic = self.signatures[col].dropna().unique()
                omic = np.concatenate([omic + modal for modal in ["_mut", "_cnv", "_rnaseq"]])
                omic = sorted(list(set(omic) & set(self.genomic_features.columns)))
                self.omic_names.append(omic)
            self.omic_sizes = [len(omic) for omic in self.omic_names]

        self.cls_ids_prep()

    def _create_patient_dict(self):
        patient_dict = {}
        for _, row in self.slide_data.iterrows():
            patient_id = row["case_id"]
            slide_id = row["slide_id"]
            if patient_id not in patient_dict:
                patient_dict[patient_id] = []
            patient_dict[patient_id].append(slide_id)
        return patient_dict

    def cls_ids_prep(self):
        self.num_classes = self.slide_data["label"].max() + 1
        self.slide_cls_ids = [[] for i in range(self.num_classes)]
        for i in range(self.num_classes):
            self.slide_cls_ids[i] = np.where(self.slide_data["label"] == i)[0]

    def getlabel(self, idx):
        return self.slide_data["label"][idx]

    def __len__(self):
        return len(self.slide_data)

    def __getitem__(self, idx):
        row = self.slide_data.iloc[idx]
        case_id, label, event_time, c = row["case_id"], row["disc_label"], row[self.label_col], row["censorship"]

        slide_id = self.patient_dict.get(case_id, [None])[0]
        if slide_id is None:
            print(f"警告: 无法为 case_id {case_id} 找到 slide_id。")
            omic_zeros = [torch.zeros(s) for s in self.omic_sizes]
            return (torch.zeros(1, 2048), [], *omic_zeros, -1, -1, -1)

        feature_filename = f"{slide_id}.pt"
        feature_path = os.path.join(self.feature_dir, feature_filename)

        try:
            cached_data = torch.load(feature_path)
            path_features = cached_data["features"]
            coords = cached_data["coords"]
        except FileNotFoundError:
            print(f"错误: 预计算的特征文件未找到: {feature_path}")
            print("请确保在开始训练前已为所有数据生成特征缓存。")
            omic_zeros = [torch.zeros(s) for s in getattr(self, "omic_sizes", [])]
            return (torch.zeros(1, 2048), [], *omic_zeros, -1, -1, -1)

        omic_data = [
            torch.tensor(self.genomic_features[og].iloc[idx].values, dtype=torch.float32) for og in self.omic_names
        ]
        return (path_features, coords, *omic_data, label, event_time, c)

    def get_scaler(self):
        return (StandardScaler().fit(self.genomic_features),)

    def apply_scaler(self, scalers: tuple = None):
        transformed = pd.DataFrame(scalers[0].transform(self.genomic_features))
        transformed.columns = self.genomic_features.columns
        self.genomic_features = transformed


# -------------------------------------------------------------------------------------
# § 6. 训练引擎
# -------------------------------------------------------------------------------------
class Engine(object):
    def __init__(self, args, results_dir):
        self.args = args
        self.results_dir = results_dir
        if args.log_data:
            from tensorboardX import SummaryWriter

            writer_dir = os.path.join(results_dir)
            if not os.path.isdir(writer_dir):
                os.mkdir(writer_dir)
            self.writer = SummaryWriter(writer_dir, flush_secs=15)
        else:
            self.writer = None
        self.best_score = 0
        self.best_epoch = 0
        self.filename_best = None

    def learning(self, model, train_loader, val_loader, criterion, optimizer, scheduler):
        if torch.cuda.is_available():
            model = model.cuda()
        if self.args.resume:
            if os.path.isfile(self.args.resume):
                print("=> loading checkpoint '{}'".format(self.args.resume))
                checkpoint = torch.load(self.args.resume)
                self.best_score = checkpoint["best_score"]
                model.load_state_dict(checkpoint["state_dict"])
                print("=> loaded checkpoint (score: {})".format(checkpoint["best_score"]))
            else:
                print("=> no checkpoint found at '{}'".format(self.args.resume))
        else:
            print("=> no checkpoint found at '{}'".format(self.args.resume))

        if self.evaluate_on_test(val_loader, model, criterion):
            return

        for epoch in range(self.args.num_epoch):
            self.epoch = epoch
            self.train(train_loader, model, criterion, optimizer)
            c_index = self.validate(val_loader, model, criterion)
            is_best = c_index > self.best_score
            if is_best:
                self.best_score = c_index
                self.best_epoch = self.epoch
                self.save_checkpoint({"epoch": epoch, "state_dict": model.state_dict(), "best_score": self.best_score})
            print(" *** best c-index={:.4f} at epoch {}".format(self.best_score, self.best_epoch))
            if scheduler is not None:
                scheduler.step()
            print(">")
        return self.best_score, self.best_epoch

    def evaluate_on_test(self, val_loader, model, criterion):
        if self.args.evaluate:
            self.validate(val_loader, model, criterion)
            return True
        return False

    def train(self, data_loader, model, criterion, optimizer):
        model.train()
        train_loss = 0.0
        all_risk_scores, all_censorships, all_event_times = [], [], []
        dataloader = tqdm(data_loader, desc="Train Epoch: {}".format(self.epoch), ncols=80)

        for batch_idx, (data_WSI, _, *omics_data, label, event_time, c) in enumerate(dataloader):
            if label.item() == -1:
                continue

            data_WSI = data_WSI.squeeze(0)
            if torch.cuda.is_available():
                data_WSI, label, c = data_WSI.cuda(), label.type(torch.LongTensor).cuda(), c.type(torch.FloatTensor).cuda()
                omics = [d.type(torch.FloatTensor).cuda() for d in omics_data]

            hazards, S, P, P_hat, G, G_hat = model(
                x_path=data_WSI,
                x_omic1=omics[0],
                x_omic2=omics[1],
                x_omic3=omics[2],
                x_omic4=omics[3],
                x_omic5=omics[4],
                x_omic6=omics[5],
            )
            sur_loss = criterion[0](hazards=hazards, S=S, Y=label, c=c)
            sim_loss_P = criterion[1](P.detach(), P_hat)
            sim_loss_G = criterion[1](G.detach(), G_hat)
            loss = sur_loss + self.args.alpha * (sim_loss_P + sim_loss_G)

            risk = -torch.sum(S, dim=1).item()
            all_risk_scores.append(risk)
            all_censorships.append(c.item())
            all_event_times.append(event_time.item())
            train_loss += loss.item()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        if len(dataloader) > 0:
            train_loss /= len(dataloader)
        else:
            train_loss = 0.0

        all_censorships = np.array(all_censorships)
        if len(all_event_times) > 0:
            try:
                c_index = concordance_index_censored(
                    (1 - all_censorships).astype(bool),
                    np.array(all_event_times),
                    np.array(all_risk_scores),
                    tied_tol=1e-08,
                )[0]
            except Exception as e:
                print(f"C-Index计算错误: {e}")
                c_index = 0.0
        else:
            c_index = 0.0

        print("loss: {:.4f}, c_index: {:.4f}".format(train_loss, c_index))
        if self.writer:
            self.writer.add_scalar("train/loss", train_loss, self.epoch)
            self.writer.add_scalar("train/c_index", c_index, self.epoch)

    def validate(self, data_loader, model, criterion):
        model.eval()
        val_loss = 0.0
        all_risk_scores, all_censorships, all_event_times = [], [], []
        dataloader = tqdm(data_loader, desc="Test Epoch: {}".format(self.epoch), ncols=80)

        for batch_idx, (data_WSI, _, *omics_data, label, event_time, c) in enumerate(dataloader):
            if label.item() == -1:
                continue

            data_WSI = data_WSI.squeeze(0)
            if torch.cuda.is_available():
                data_WSI, label, c = data_WSI.cuda(), label.type(torch.LongTensor).cuda(), c.type(torch.FloatTensor).cuda()
                omics = [d.type(torch.FloatTensor).cuda() for d in omics_data]

            with torch.no_grad():
                hazards, S, P, P_hat, G, G_hat = model(
                    x_path=data_WSI,
                    x_omic1=omics[0],
                    x_omic2=omics[1],
                    x_omic3=omics[2],
                    x_omic4=omics[3],
                    x_omic5=omics[4],
                    x_omic6=omics[5],
                )

            sur_loss = criterion[0](hazards=hazards, S=S, Y=label, c=c)
            sim_loss_P = criterion[1](P.detach(), P_hat)
            sim_loss_G = criterion[1](G.detach(), G_hat)
            loss = sur_loss + self.args.alpha * (sim_loss_P + sim_loss_G)

            risk = -torch.sum(S, dim=1).item()
            all_risk_scores.append(risk)
            all_censorships.append(c.item())
            all_event_times.append(event_time.item())
            val_loss += loss.item()

        if len(dataloader) > 0:
            val_loss /= len(dataloader)
        else:
            val_loss = 0.0

        all_censorships = np.array(all_censorships)

        print(f"\n--- 验证集详细信息 ---")
        if len(all_event_times) > 0:
            try:
                c_index = concordance_index_censored(
                    (1 - all_censorships).astype(bool),
                    np.array(all_event_times),
                    np.array(all_risk_scores),
                    tied_tol=1e-08,
                )[0]
            except Exception as e:
                print(f"验证集C-Index计算错误: {e}")
                c_index = 0.0
        else:
            print("验证集为空，无法计算c-index。")
            print("-----------------------\n")
            c_index = 0.0

        print("loss: {:.4f}, c_index: {:.4f}".format(val_loss, c_index))
        if self.writer:
            self.writer.add_scalar("val/loss", val_loss, self.epoch)
            self.writer.add_scalar("val/c-index", c_index, self.epoch)
        return c_index

    def save_checkpoint(self, state):
        if self.filename_best is not None and os.path.exists(self.filename_best):
            os.remove(self.filename_best)
        self.filename_best = os.path.join(
            self.results_dir,
            "model_best_{score:.4f}_{epoch}.pth.tar".format(score=state["best_score"], epoch=state["epoch"]),
        )
        print("save best model {filename}".format(filename=self.filename_best))
        torch.save(state, self.filename_best)


# -------------------------------------------------------------------------------------
# § 7. 主执行逻辑
# -------------------------------------------------------------------------------------
def create_attention_heatmap(slide_path, coords, attention_scores, args, output_path):
    print("正在生成高质量的注意力热图...")
    try:
        slide = openslide.OpenSlide(slide_path)
    except openslide.OpenSlideError:
        print(f"错误: 无法打开 SVS 文件 {slide_path}")
        return

    thumb_level = slide.level_count - 1
    thumb_w, thumb_h = slide.level_dimensions[thumb_level]
    thumbnail = slide.read_region((0, 0), thumb_level, (thumb_w, thumb_h)).convert("RGB")
    thumbnail_np = np.array(thumbnail)

    if not coords:
        print("警告: 坐标列表为空，无法生成热图。")
        slide.close()
        return

    src_level = args.patch_level
    if src_level >= slide.level_count:
        src_level = slide.level_count - 1
        print(f"警告: patch_level 超出范围，自动调整为: {src_level}")

    scale_factor = slide.level_downsamples[src_level] / slide.level_downsamples[thumb_level]
    patch_thumb_w = max(1, int(args.patch_size / scale_factor))
    patch_thumb_h = max(1, int(args.patch_size / scale_factor))

    heatmap = np.zeros((thumb_h, thumb_w), dtype=np.float32)
    scores = np.array(attention_scores).flatten()
    if scores.max() == scores.min():
        scores_norm = np.zeros_like(scores)
        print("警告: 所有注意力分数都相同，热图将为空。")
    else:
        scores_norm = (scores - scores.min()) / (scores.max() - scores.min())

    for (x, y), score in zip(coords, scores_norm):
        thumb_x = int(x / scale_factor)
        thumb_y = int(y / scale_factor)
        y_start, y_end = thumb_y, thumb_y + patch_thumb_h
        x_start, x_end = thumb_x, thumb_x + patch_thumb_w
        if y_end <= thumb_h and x_end <= thumb_w:
            heatmap[y_start:y_end, x_start:x_end] = score

    kernel_size = (51, 51)
    sigma = max(patch_thumb_w * 2, 5)
    heatmap = cv2.GaussianBlur(heatmap, kernel_size, sigmaX=sigma, sigmaY=sigma)

    if heatmap.max() > 0:
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())

    heatmap_uint8 = (heatmap * 255).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    alpha, beta = 0.5, 0.5
    overlay = cv2.addWeighted(heatmap_color, alpha, thumbnail_np, beta, 0)

    tissue_mask, _ = get_tissue_mask(slide, level=thumb_level)
    tissue_mask_3c = np.stack([tissue_mask] * 3, axis=-1)

    if tissue_mask_3c.shape != overlay.shape:
        overlay = cv2.resize(overlay, (tissue_mask_3c.shape[1], tissue_mask_3c.shape[0]))
        thumbnail_np = cv2.resize(thumbnail_np, (tissue_mask_3c.shape[1], tissue_mask_3c.shape[0]))

    final_image = np.where(tissue_mask_3c, overlay, thumbnail_np)

    output_dir = os.path.dirname(output_path)
)

    plt.figure(figsize=(12, 12), dpi=300)
    plt.imshow(final_image)
    plt.axis("off")
    plt.title(f"Attention Heatmap: {os.path.basename(slide_path)}", fontsize=10)
    plt.tight_layout(pad=0)
    plt.savefig(output_path, bbox_inches="tight", pad_inches=0)
    plt.close()
    slide.close()

    print(f"高质量注意力热图已保存至: {output_path}")


def main(args):
    set_seed(args.seed)

    print("正在准备数据集...")
    print(f"扫描SVS文件目录: {args.data_root_dir}")
    if not os.path.isdir(args.data_root_dir):
        raise FileNotFoundError(f"SVS数据目录不存在: {args.data_root_dir}")

    svs_files = [f for f in os.listdir(args.data_root_dir) if f.lower().endswith(".svs")]
    if not svs_files:
        raise FileNotFoundError(f"在目录 {args.data_root_dir} 中没有找到任何 .svs 文件。")
    print(f"发现 {len(svs_files)} 个SVS文件。")

    csv_path = f"./csv/{args.dataset}_all_clean.csv"
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"未找到元数据CSV文件: {csv_path}。")
    all_csv_data = pd.read_csv(csv_path, low_memory=False)

    all_csv_data["slide_filename"] = all_csv_data["slide_id"].apply(
        lambda x: x if str(x).lower().endswith(".svs") else f"{x}.svs"
    )

    all_available_data = all_csv_data[all_csv_data["slide_filename"].isin(svs_files)].copy()
    if all_available_data.empty:
        print("警告: CSV中的slide_id与SVS文件名直接匹配失败，尝试仅匹配ID...")
        svs_ids = [f.split(".")[0] for f in svs_files]
        all_available_data = all_csv_data[all_csv_data["slide_id"].isin(svs_ids)].copy()

        if all_available_data.empty:
            raise ValueError("SVS文件与CSV中的slide_id无法匹配。请检查文件名是否一致。")

    print(f"成功将 SVS文件与CSV中的 {len(all_available_data)} 条记录匹配。")

    print("\n" + "=" * 50)
    print("开始检查并生成特征缓存...")
    print(f"特征将被保存到: {args.feature_dir}")
    if not os.path.exists(args.feature_dir):
        os.makedirs(args.feature_dir)
        print(f"已创建特征目录: {args.feature_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feature_extractor = ResNet_Feature_Extractor().to(device)
    feature_extractor.eval()
    preprocess = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    for _, row in tqdm(all_available_data.iterrows(), total=len(all_available_data), desc="检查/缓存特征", ncols=80):
        slide_id = row["slide_id"]
        slide_filename = row["slide_filename"]

        if not os.path.exists(os.path.join(args.data_root_dir, slide_filename)):
            possible_names = [f for f in svs_files if f.startswith(slide_id)]
            if possible_names:
                slide_filename = possible_names[0]
            else:
                continue

        svs_path = os.path.join(args.data_root_dir, slide_filename)
        feature_filename = f"{slide_id}.pt"
        feature_path = os.path.join(args.feature_dir, feature_filename)

        if not os.path.exists(feature_path):
            if not os.path.exists(svs_path):
                print(f"警告: SVS文件 {svs_path} 不存在，无法生成特征。跳过 {slide_id}。")
                continue
            extract_and_save_features_for_slide(svs_path, feature_extractor, preprocess, args, feature_path)

    print("所有必需的特征均已缓存。")
    print("=" * 50 + "\n")
    del feature_extractor
    torch.cuda.empty_cache()

    uncensored_df = all_available_data[all_available_data["censorship"] < 1]
    if len(uncensored_df) < 4:
        print("未审查样本过少，无法使用四分位数，使用平均值切割")
        mean_val = all_available_data["survival_months"].mean()
        disc_labels = (all_available_data["survival_months"] > mean_val).astype(int)
        all_available_data["disc_label"] = disc_labels
    else:
        _, q_bins = pd.qcut(uncensored_df["survival_months"], q=4, retbins=True, labels=False)
        q_bins[-1] = all_available_data["survival_months"].max() + 1e-6
        q_bins[0] = all_available_data["survival_months"].min() - 1e-6

        disc_labels, _ = pd.cut(
            all_available_data["survival_months"],
            bins=q_bins,
            retbins=True,
            labels=False,
            right=False,
            include_lowest=True,
        )
        all_available_data["disc_label"] = disc_labels.values.astype(int)

    key_count = 0
    label_dict = {}
    for i in range(4):
        for c in [0, 1]:
            label_dict.update({(i, c): key_count})
            key_count += 1

    all_available_data["disc_label"] = all_available_data["disc_label"].fillna(0).astype(int)
    all_available_data["label"] = all_available_data.apply(
        lambda row: label_dict.get((row["disc_label"], int(row["censorship"])), -1), axis=1
    )

    n_splits = 5
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=args.seed)
    fold_results = []

    last_fold_best_model_path = None
    last_fold_val_dataset = None
    final_train_dataset_for_vis = None

    try:
        splits = list(skf.split(all_available_data, all_available_data["label"]))
    except ValueError:
        print("警告: 类样本过少无法分层，切换到KFold")
        from sklearn.model_selection import KFold

        kf = KFold(n_splits=n_splits, shuffle=True, random_state=args.seed)
        splits = list(kf.split(all_available_data))

    for fold, (train_idx, val_idx) in enumerate(splits):
        print("\n" + "=" * 50)
        print(f"=============== FOLD {fold + 1}/{n_splits} ===============")
        print("=" * 50)

        train_df = all_available_data.iloc[train_idx]
        val_df = all_available_data.iloc[val_idx]

        signatures_path = "./csv/signatures.csv"
        if os.path.exists(signatures_path):
            signatures = pd.read_csv(signatures_path)
            apply_sig = True
        else:
            print("未找到 signatures.csv，将不使用基因组签名特征。")
            signatures = None
            apply_sig = False

        train_dataset = Survival_SVS_Dataset(df=train_df, feature_dir=args.feature_dir, args=args, apply_sig=apply_sig, signatures=signatures)
        val_dataset = Survival_SVS_Dataset(df=val_df, feature_dir=args.feature_dir, args=args, apply_sig=apply_sig, signatures=signatures)
        final_train_dataset_for_vis = train_dataset

        print("正在进行数据标准化...")
        try:
            scalers = train_dataset.get_scaler()
            train_dataset.apply_scaler(scalers)
            val_dataset.apply_scaler(scalers)
        except ValueError:
            print("基因组特征为空或全为常量，跳过标准化。")

        train_loader = get_split_loader(train_dataset, training=True, weighted=args.weighted_sample, batch_size=args.batch_size)
        val_loader = get_split_loader(val_dataset, training=False, batch_size=args.batch_size)

        print("正在为当前折构建新模型...")
        if args.use_innovation:
            print("【注意】已启用三模块协同创新套件：SGAP + DSEP + ARFH。")
        else:
            print("【注意】创新模块已关闭，使用标准多尺度卷积位置编码。")

        model_dict = {
            "omic_sizes": train_dataset.omic_sizes if hasattr(train_dataset, "omic_sizes") else [],
            "n_classes": 4,
            "fusion": args.fusion,
            "model_size": args.model_size,
            "use_innovation": args.use_innovation,
        }
        model = CMTA(**model_dict)
        criterion = define_loss(args)
        optimizer = define_optimizer(args, model)
        scheduler = define_scheduler(args, optimizer)

        fold_results_dir = "./results/{dataset}/[{model}]-fold_{fold}-[{time}]".format(
            dataset=args.dataset,
            model=args.model,
            fold=fold + 1,
            time=time.strftime("%Y-%m-%d_%H-%M-%S"),
        )
        if not os.path.exists(fold_results_dir):
            os.makedirs(fold_results_dir)

        engine = Engine(args, fold_results_dir)

        print(f"开始训练第 {fold + 1} 折...")
        best_score, _ = engine.learning(model, train_loader, val_loader, criterion, optimizer, scheduler)
        fold_results.append(best_score)

        if fold == n_splits - 1:
            last_fold_best_model_path = engine.filename_best
            last_fold_val_dataset = val_dataset

    print("\n" + "=" * 50)
    print("交叉验证全部完成。")
    print(f"每折的最佳 C-Index: {[round(score, 4) for score in fold_results]}")
    if fold_results:
        print(f"平均 C-Index: {np.mean(fold_results):.4f} ± {np.std(fold_results):.4f}")
    print("=" * 50 + "\n")

    if last_fold_best_model_path and os.path.exists(last_fold_best_model_path):
        print("使用最后一折的最佳模型进行推理和可视化...")

        model_dict = {
            "omic_sizes": final_train_dataset_for_vis.omic_sizes if hasattr(final_train_dataset_for_vis, "omic_sizes") else [],
            "n_classes": 4,
            "fusion": args.fusion,
            "model_size": args.model_size,
            "use_innovation": args.use_innovation,
        }
        vis_model = CMTA(**model_dict).cuda()
        checkpoint = torch.load(last_fold_best_model_path)
        vis_model.load_state_dict(checkpoint["state_dict"])
        vis_model.eval()

        if len(last_fold_val_dataset) == 0:
            print("最后一折的验证集为空，无法进行可视化。")
            return

        vis_idx = -1
        for i in range(len(last_fold_val_dataset)):
            if last_fold_val_dataset.getlabel(i) != -1:
                vis_idx = i
                break

        if vis_idx == -1:
            print("最后一折的验证集中没有找到有效样本进行可视化。")
            return

        (path_features, coords, *omics_data, label, event_time, c) = last_fold_val_dataset[vis_idx]

        slide_id_to_vis = last_fold_val_dataset.slide_data.iloc[vis_idx]["slide_id"]
        print(f"正在对样本进行可视化: {slide_id_to_vis}")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        path_features = path_features.to(device)
        omics = [d.unsqueeze(0).to(device) for d in omics_data]

        with torch.no_grad():
            outputs = vis_model(
                return_attn=True,
                x_path=path_features,
                x_omic1=omics[0],
                x_omic2=omics[1],
                x_omic3=omics[2],
                x_omic4=omics[3],
                x_omic5=omics[4],
                x_omic6=omics[5],
            )
        attn_weights = outputs[-1]

        attn_scores = attn_weights[0, :, 0, 1:].mean(dim=0).cpu().numpy()

        slide_filename_to_vis = f"{slide_id_to_vis}.svs" if not str(slide_id_to_vis).lower().endswith(".svs") else slide_id_to_vis
        svs_path_to_vis = os.path.join(args.data_root_dir, slide_filename_to_vis)

        if not os.path.exists(svs_path_to_vis):
            print(f"无法找到SVS文件 {svs_path_to_vis}，跳过热力图生成。")
        else:
            main_results_dir = "./results/{dataset}/[{model}]-CV_Summary".format(dataset=args.dataset, model=args.model)
            if not os.path.exists(main_results_dir):
                os.makedirs(main_results_dir)
            heatmap_output_path = os.path.join(main_results_dir, f"attention_heatmap_{slide_id_to_vis}.png")

            create_attention_heatmap(svs_path_to_vis, coords, attn_scores, args, heatmap_output_path)


if __name__ == "__main__":
    import torch.multiprocessing as mp

    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    args = parse_args()
    main(args)
    print("Finished!")
