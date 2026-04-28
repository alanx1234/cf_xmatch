import numpy as np
import torch
from torch import Tensor
from sklearn.preprocessing import StandardScaler
import zuko

from .constants import LOGA_GRID


def age_posterior(flow:        zuko.flows.NSF,
                  log_prot:    float,
                  c_fixed:     np.ndarray,
                  age_col_idx: int,
                  scaler:      StandardScaler,
                  loga_grid:   np.ndarray = LOGA_GRID
                  ) -> np.ndarray:
    """Evaluate P(log_prot | age, ...) over loga_grid for one star.

    c_fixed holds all conditioning vars in their original (un-normalized) values,
    with the age position (age_col_idx) set to a placeholder — it gets replaced
    at each grid point. Each full conditioning vector is normalized via scaler
    before being passed to the flow.

    No mass prior needed — P(m | sigma_m, age) is constant over age on the MS
    and cancels after normalization. Age prior is uniform and also cancels.

    Returns normalized probability array over loga_grid.
    """
    log_probs = np.zeros(len(loga_grid))

    with torch.no_grad():
        x = torch.tensor([[log_prot]], dtype=torch.float32)

        for i, loga in enumerate(loga_grid):
            c_raw             = c_fixed.copy()
            c_raw[age_col_idx] = loga
            c_norm            = scaler.transform(c_raw.reshape(1, -1)).astype(np.float32)
            c_t               = torch.tensor(c_norm)
            log_probs[i]      = flow(c_t).log_prob(x).item()

    probs = np.exp(log_probs - log_probs.max())  # numerical stability
    probs /= np.trapz(probs, loga_grid)
    return probs


def batch_posteriors(flow:      zuko.flows.NSF,
                     df:        object,
                     obs_col:   str,
                     cond_cols: list[str],
                     scaler:    StandardScaler,
                     loga_grid: np.ndarray = LOGA_GRID
                     ) -> np.ndarray:
    """Vectorized age posteriors for all stars in df.

    Builds an (N x len(loga_grid), n_cond) conditioning tensor in one pass
    by tiling each star's fixed conditioning vars across the age grid, then
    replacing the age column with grid values. Normalized per star.

    Returns array of shape (N, len(loga_grid)).
    """
    age_col_idx = cond_cols.index('log_age_myr')
    n_stars     = len(df)
    n_grid      = len(loga_grid)

    c_raw = df[cond_cols].values.astype(np.float32)       # (N, n_cond)
    x_raw = df[obs_col].values.astype(np.float32)         # (N,)

    # Tile: repeat each star's conditioning n_grid times
    c_tiled = np.repeat(c_raw, n_grid, axis=0)            # (N*n_grid, n_cond)
    c_tiled[:, age_col_idx] = np.tile(loga_grid, n_stars) # replace age column

    c_norm = scaler.transform(c_tiled).astype(np.float32) # (N*n_grid, n_cond)
    x_tile = np.repeat(x_raw, n_grid).astype(np.float32)  # (N*n_grid,)

    c_t = torch.tensor(c_norm)
    x_t = torch.tensor(x_tile).unsqueeze(1)

    with torch.no_grad():
        log_probs = flow(c_t).log_prob(x_t).numpy()       # (N*n_grid,)

    log_probs = log_probs.reshape(n_stars, n_grid)

    # Normalize each star's posterior
    log_probs -= log_probs.max(axis=1, keepdims=True)
    probs      = np.exp(log_probs)
    probs     /= np.trapz(probs, loga_grid, axis=1)[:, None]
    return probs
