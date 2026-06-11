import numpy as np
import torch
from torch import Tensor
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import OneCycleLR
import zuko

from .constants import W_F, PRIOR_LOGPROT


class _CombinedLRScheduler:
    """Exponential decay + cosine annealing scheduler from ChronoFlow.

    Stepped once per epoch (not per batch) to preserve the original schedule shape.
    """

    def __init__(self, optimizer, T_max: int, initial_lr: float, decay_rate: float):
        self.optimizer     = optimizer
        self.T_max         = T_max
        self.initial_lr    = initial_lr
        self.decay_rate    = decay_rate
        self.current_epoch = 0

    def step(self):
        for pg in self.optimizer.param_groups:
            pg['lr'] = (self.initial_lr
                        * (self.decay_rate ** self.current_epoch)
                        * 0.5 * (1 + np.cos(np.pi * self.current_epoch / self.T_max)))
        self.current_epoch += 1


def _draw_age_samples(
    c_b:        Tensor,
    log_age_b:  Tensor,
    sigma_lo_b: Tensor,
    sigma_hi_b: Tensor,
    has_err_b:  Tensor,
    age_mean:   float,
    age_std:    float,
    n_samples:  int = 10,
) -> Tensor:
    """Draw n_samples age perturbations per star; return c_expanded (B*n_samples, n_cond).

    Age column (index 0) is replaced with normalized sampled ages. Stars without
    age_err (has_err=0) keep their point age across all samples (sigma=0).
    Asymmetric Gaussian: sigma_lo applied when eps<0, sigma_hi when eps>=0.
    """
    B   = c_b.shape[0]
    eps = torch.randn(B, n_samples, device=c_b.device)

    sigma = torch.where(
        eps < 0,
        sigma_lo_b.unsqueeze(1).expand(-1, n_samples),
        sigma_hi_b.unsqueeze(1).expand(-1, n_samples),
    )
    sigma = sigma * has_err_b.unsqueeze(1)  # zero out for stars without age_err

    log_age_s = (log_age_b.unsqueeze(1) + eps * sigma).clamp(0.0, 4.14)  # (B, n_samples)
    age_norm  = (log_age_s - age_mean) / age_std

    c_exp        = c_b.unsqueeze(1).expand(-1, n_samples, -1).clone()  # (B, n_samples, n_cond)
    c_exp[:, :, 0] = age_norm
    return c_exp.reshape(B * n_samples, -1)


def _draw_age_mass_samples(
    c_b:           Tensor,
    log_age_b:     Tensor,
    age_sig_lo:    Tensor,
    age_sig_hi:    Tensor,
    age_has_err:   Tensor,
    mass_b:        Tensor,
    mass_sig_lo:   Tensor,
    mass_sig_hi:   Tensor,
    mass_has_err:  Tensor,
    age_mean:      float,
    age_std:       float,
    mass_mean:     float,
    mass_std:      float,
    age_col_idx:   int   = 0,
    mass_col_idx:  int   = 1,
    n_samples:     int   = 10,
    mass_floor:    float = 0.05,
) -> Tensor:
    """Joint age+mass perturbation. Independent asymmetric Gaussian draws per axis.

    Used when sample_mass=True. Mass perturbations operate in linear M_sun units;
    the mass column gets renormalized using scaler stats. Mass is floored at
    mass_floor M_sun to prevent unphysical negative samples in the lower tail.
    """
    B = c_b.shape[0]

    # Age
    eps_age = torch.randn(B, n_samples, device=c_b.device)
    sig_age = torch.where(eps_age < 0,
                          age_sig_lo.unsqueeze(1).expand(-1, n_samples),
                          age_sig_hi.unsqueeze(1).expand(-1, n_samples)) * age_has_err.unsqueeze(1)
    log_age_s = (log_age_b.unsqueeze(1) + eps_age * sig_age).clamp(0.0, 4.14)
    age_norm  = (log_age_s - age_mean) / age_std

    # Mass
    eps_mass = torch.randn(B, n_samples, device=c_b.device)
    sig_mass = torch.where(eps_mass < 0,
                           mass_sig_lo.unsqueeze(1).expand(-1, n_samples),
                           mass_sig_hi.unsqueeze(1).expand(-1, n_samples)) * mass_has_err.unsqueeze(1)
    mass_s    = (mass_b.unsqueeze(1) + eps_mass * sig_mass).clamp(min=mass_floor)
    mass_norm = (mass_s - mass_mean) / mass_std

    c_exp                          = c_b.unsqueeze(1).expand(-1, n_samples, -1).clone()
    c_exp[:, :, age_col_idx]       = age_norm
    c_exp[:, :, mass_col_idx]      = mass_norm
    return c_exp.reshape(B * n_samples, -1)


def _compute_loss(
    flow:      zuko.flows.NSF,
    x_b:       Tensor,
    c_exp:     Tensor,
    n_samples: int,
    ln_p_out:  float,
    loss_agg:  str = 'mean',
) -> Tensor:
    """Multi-sample ChronoFlow loss.

    x_b: (B, 1), c_exp: (B*n_samples, n_cond).

    loss_agg controls how per-sample log-probs are combined into a per-star score:
      - 'mean'      : (1/N) Σ log P_i — average NLL across samples. Default. Matches
                      Phil's literal spec; wins on test LL + residuals in loss_comparison.ipynb.
      - 'logsumexp' : log((1/N) Σ P_i) — proper MC marginal likelihood. Slightly better
                      coverage calibration but worse held-out fit.
    """
    x_exp     = x_b.repeat_interleave(n_samples, dim=0)          # (B*n_samples, 1)
    lp        = flow(c_exp).log_prob(x_exp)                       # (B*n_samples,)
    lp_rs     = lp.reshape(-1, n_samples)
    if loss_agg == 'logsumexp':
        lp_star = lp_rs.logsumexp(dim=1) - np.log(n_samples)      # (B,)
    elif loss_agg == 'mean':
        lp_star = lp_rs.mean(dim=1)                               # (B,)
    else:
        raise ValueError(f"loss_agg must be 'logsumexp' or 'mean', got {loss_agg!r}")
    ln_p_flow = lp_star + np.log(W_F)
    ln_p_bg   = torch.full_like(ln_p_flow, ln_p_out)
    return -torch.stack([ln_p_flow, ln_p_bg]).logsumexp(dim=0).mean()


def train_fold(
    flow:         zuko.flows.NSF,
    x_train:      Tensor,
    c_train:      Tensor,
    log_age_tr:   Tensor,
    sigma_lo_tr:  Tensor,
    sigma_hi_tr:  Tensor,
    has_err_tr:   Tensor,
    scaler:       object,
    x_val:        Tensor | None = None,
    c_val:        Tensor | None = None,
    log_age_val:  Tensor | None = None,
    sigma_lo_val: Tensor | None = None,
    sigma_hi_val: Tensor | None = None,
    has_err_val:  Tensor | None = None,
    total_steps:  int   = 5000,
    lr:           float = 1e-3,
    batch_size:   int   = 256,
    n_samples:    int   = 10,
    use_onecycle: bool  = False,
    prior_bounds: tuple = PRIOR_LOGPROT,
    print_every:  int   = 100,
    early_stop:   bool  = True,
    # Optional mass MC sampling
    sample_mass:           bool          = False,
    mass_tr:               Tensor | None = None,
    mass_sig_lo_tr:        Tensor | None = None,
    mass_sig_hi_tr:        Tensor | None = None,
    mass_has_err_tr:       Tensor | None = None,
    mass_val:              Tensor | None = None,
    mass_sig_lo_val:       Tensor | None = None,
    mass_sig_hi_val:       Tensor | None = None,
    mass_has_err_val:      Tensor | None = None,
    mass_col_idx:          int           = 1,
    loss_agg:              str           = 'mean',
) -> tuple[list[float], list[float]]:
    """Train flow on one fold with mini-batches, multi-sample age uncertainty, val loss tracking.

    Epochs are derived from total_steps // steps_per_epoch so that gradient update
    count stays fixed regardless of batch size.
    Returns (train_curve, val_curve) — one value per epoch.
    CombinedLR is stepped once per epoch; OneCycleLR is stepped once per batch.
    Set early_stop=False and omit val tensors to train on the full dataset for a fixed
    number of steps (used for final model training after architecture selection).
    """
    has_val = x_val is not None
    if early_stop and not has_val:
        raise ValueError('early_stop=True requires val tensors to be provided')

    ln_p_out = float(np.log((1 - W_F) / (prior_bounds[1] - prior_bounds[0])))
    age_mean = float(scaler.mean_[0])
    age_std  = float(scaler.scale_[0])

    if sample_mass:
        if mass_tr is None or mass_sig_lo_tr is None or mass_sig_hi_tr is None or mass_has_err_tr is None:
            raise ValueError('sample_mass=True requires mass_tr + mass_sig_lo_tr + mass_sig_hi_tr + mass_has_err_tr')
        mass_mean = float(scaler.mean_[mass_col_idx])
        mass_std  = float(scaler.scale_[mass_col_idx])

    optimizer = torch.optim.Adam(flow.parameters(), lr=lr)

    # Seed everything before any random draws for full reproducibility
    torch.manual_seed(19)

    # Pre-draw fixed perturbed samples once per fold — same n_samples used across all epochs
    N = len(x_train)
    if sample_mass:
        c_train_exp = _draw_age_mass_samples(
            c_train, log_age_tr, sigma_lo_tr, sigma_hi_tr, has_err_tr,
            mass_tr, mass_sig_lo_tr, mass_sig_hi_tr, mass_has_err_tr,
            age_mean, age_std, mass_mean, mass_std,
            mass_col_idx=mass_col_idx, n_samples=n_samples,
        )
    else:
        c_train_exp = _draw_age_samples(c_train, log_age_tr, sigma_lo_tr, sigma_hi_tr,
                                        has_err_tr, age_mean, age_std, n_samples)
    c_train_samples = c_train_exp.reshape(N, n_samples, -1)  # (N, n_samples, n_cond)

    if has_val:
        if sample_mass:
            c_val_exp = _draw_age_mass_samples(
                c_val, log_age_val, sigma_lo_val, sigma_hi_val, has_err_val,
                mass_val, mass_sig_lo_val, mass_sig_hi_val, mass_has_err_val,
                age_mean, age_std, mass_mean, mass_std,
                mass_col_idx=mass_col_idx, n_samples=n_samples,
            )
        else:
            c_val_exp = _draw_age_samples(c_val, log_age_val, sigma_lo_val, sigma_hi_val,
                                          has_err_val, age_mean, age_std, n_samples)
        # c_val_exp: (M*n_samples, n_cond) — fixed, reused every epoch for deterministic val loss

    # Seeded generator for reproducible DataLoader shuffling
    g = torch.Generator()
    g.manual_seed(19)
    dataset      = TensorDataset(x_train, torch.arange(N))
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=g)
    steps_per_epoch = len(train_loader)
    epochs          = max(1, total_steps // steps_per_epoch)
    print(f'  total_steps={total_steps}  steps_per_epoch={steps_per_epoch}  epochs={epochs}')

    if use_onecycle:
        scheduler = OneCycleLR(
            optimizer,
            max_lr           = lr,
            total_steps      = epochs * steps_per_epoch,
            pct_start        = 0.1,
            div_factor       = 1e2,
            final_div_factor = 1e2,
        )
    else:
        scheduler = _CombinedLRScheduler(
            optimizer,
            T_max      = 1000,
            initial_lr = lr,
            decay_rate = 10 ** (-4 / 6000),
        )

    train_curve:  list[float] = []
    val_curve:    list[float] = []
    best_val_loss = float('inf')
    best_state    = None

    for epoch in range(epochs):
        flow.train()
        batch_losses = []

        for x_b, idx_b in train_loader:
            c_exp_b = c_train_samples[idx_b].reshape(-1, c_train.shape[1])  # (B*n_samples, n_cond)
            loss    = _compute_loss(flow, x_b, c_exp_b, n_samples, ln_p_out, loss_agg=loss_agg)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if use_onecycle:
                scheduler.step()
            batch_losses.append(loss.item())

        if not use_onecycle:
            scheduler.step()

        train_curve.append(float(np.mean(batch_losses)))

        if has_val:
            flow.eval()
            with torch.no_grad():
                val_loss = _compute_loss(flow, x_val, c_val_exp, n_samples, ln_p_out, loss_agg=loss_agg)
            val_curve.append(val_loss.item())

            if early_stop and val_loss.item() < best_val_loss:
                best_val_loss = val_loss.item()
                best_state    = {k: v.clone() for k, v in flow.state_dict().items()}

        if (epoch + 1) % print_every == 0:
            if has_val:
                print(f'epoch {epoch+1:5d}/{epochs}  '
                      f'train {train_curve[-1]:.6f}  val {val_curve[-1]:.6f}')
            else:
                print(f'epoch {epoch+1:5d}/{epochs}  train {train_curve[-1]:.6f}')

    if early_stop and best_state is not None:
        flow.load_state_dict(best_state)
        print(f'  → best checkpoint restored  (val {best_val_loss:.6f})')
    return train_curve, val_curve


def compute_test_ll(
    flow:   zuko.flows.NSF,
    x_test: Tensor,
    c_test: Tensor,
) -> float:
    """Average log likelihood on test set at true conditioning values.

    Evaluates p(P_rot | age_true, mass) for each test star — no age sampling,
    consistent with inference where age is the unknown and not available.
    Returns mean log_prob over all test stars (higher = better, closer to 0).
    """
    flow.eval()
    with torch.no_grad():
        return flow(c_test).log_prob(x_test).mean().item()
