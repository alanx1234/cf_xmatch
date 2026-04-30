import numpy as np
import torch
from torch import Tensor
from typing import Callable
import zuko

from .constants import W_F, PRIOR_LOGPROT, PRIOR_LOGROSSBY
from .data import make_tensors, sample_log_age


class _CombinedLRScheduler:
    """Exponential decay + cosine annealing scheduler from ChronoFlow."""

    def __init__(self, optimizer, T_max: int, initial_lr: float, decay_rate: float):
        self.optimizer    = optimizer
        self.T_max        = T_max
        self.initial_lr   = initial_lr
        self.decay_rate   = decay_rate
        self.current_epoch = 0

    def step(self):
        for pg in self.optimizer.param_groups:
            pg['lr'] = (self.initial_lr
                        * (self.decay_rate ** self.current_epoch)
                        * 0.5 * (1 + np.cos(np.pi * self.current_epoch / self.T_max)))
        self.current_epoch += 1


def train_fold(flow:           zuko.flows.NSF,
               x:              Tensor,
               c:              Tensor,
               obs_col:        str              = 'log_prot',
               steps:          int              = 5000,
               lr:             float            = 1e-3,
               age_weights:    Tensor | None    = None,
               age_sample_fn:  Callable | None  = None,
               age_df:         object           = None,
               age_col:        str | None       = None,
               err_lo_col:     str | None       = None,
               err_hi_col:     str | None       = None,
               cond_cols:      list[str] | None = None,
               scaler:         object           = None,
               prior_bounds:   tuple            = PRIOR_LOGPROT,
               print_every:    int              = 1000,
               ) -> list[float]:
    """Train flow on one fold. Returns loss curve.

    1A: age_weights=None, age_sample_fn=None.
    1B: pass age_weights tensor (inverse age uncertainty, normalized).
        age_weights are applied to ln_p_cond before W_F scaling and the
        background mixture, matching the original ChronoFlow formulation.
    1C: pass age_sample_fn=sample_log_age plus age_df, age_col, err_lo_col,
        err_hi_col, cond_cols, scaler so sampling occurs before normalization
        each step.
    obs_col is reused for 1C resampling so sampled-age models can target either
    log_prot or a derived observable such as log_rossby.
    Pass prior_bounds=PRIOR_LOGROSSBY for Rossby number models.
    """
    ln_p_out = np.log(
        (1 - W_F) / (prior_bounds[1] - prior_bounds[0])
    )

    optimizer = torch.optim.Adam(flow.parameters(), lr=lr)
    scheduler = _CombinedLRScheduler(
        optimizer,
        T_max      = 1000,
        initial_lr = lr,
        decay_rate = 10 ** (-4 / 6000),
    )

    torch.manual_seed(19)
    loss_curve = []

    for step in range(steps + 1):
        # 1C: resample log_age each step before normalization
        if age_sample_fn is not None:
            sampled_df    = age_sample_fn(age_df, age_col, err_lo_col, err_hi_col)
            _, c, _       = make_tensors(sampled_df, obs_col, cond_cols, scaler)

        ln_p_cond = flow(c).log_prob(x)

        # 1B: scale per-star log-prob by inverse age uncertainty weight
        if age_weights is not None:
            ln_p_cond = age_weights * ln_p_cond

        ln_p_flow = W_F * ln_p_cond
        ln_p_bg   = torch.full_like(ln_p_flow, ln_p_out)
        loss      = -torch.stack([ln_p_flow, ln_p_bg], dim=0).logsumexp(dim=0).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        loss_curve.append(loss.item())
        if step % print_every == 0:
            print(f'step {step:5d}  loss {loss.item():.6f}')

    return loss_curve
