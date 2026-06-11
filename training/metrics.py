import numpy as np
import pandas as pd

from .constants import LOGA_GRID


def compute_residuals(df:        pd.DataFrame,
                      posteriors: np.ndarray,
                      loga_grid:  np.ndarray = LOGA_GRID
                      ) -> pd.DataFrame:
    """Returns df with added columns: p16, p50, p84, inferred_age_gyr, residual_dex.

    p16/p50/p84 computed via cumulative trapezoid integration over the posterior.
    residual_dex = log10(inferred_age_myr) - log10(true_age_myr)
                 = inferred p50 on loga_grid minus true log_age_myr.
    """
    df = df.copy()
    cdf = np.cumsum(posteriors * np.gradient(loga_grid), axis=1)
    cdf /= cdf[:, -1:]  # normalize each row to [0, 1]

    p16 = np.array([np.interp(0.16, cdf[i], loga_grid) for i in range(len(df))])
    p50 = np.array([np.interp(0.50, cdf[i], loga_grid) for i in range(len(df))])
    p84 = np.array([np.interp(0.84, cdf[i], loga_grid) for i in range(len(df))])

    df['p16']              = p16
    df['p50']              = p50
    df['p84']              = p84
    df['inferred_age_gyr'] = 10 ** p50 / 1000
    df['residual_dex']     = p50 - df['log_age_myr'].values
    return df


def compute_precision(posteriors: np.ndarray,
                      loga_grid:  np.ndarray = LOGA_GRID
                      ) -> np.ndarray:
    """Returns p84 - p16 in dex for each posterior."""
    cdf = np.cumsum(posteriors * np.gradient(loga_grid), axis=1)
    cdf /= cdf[:, -1:]

    p16 = np.array([np.interp(0.16, cdf[i], loga_grid) for i in range(len(posteriors))])
    p84 = np.array([np.interp(0.84, cdf[i], loga_grid) for i in range(len(posteriors))])
    return p84 - p16


def compute_coverage(posteriors:   np.ndarray,
                     log_age_true: np.ndarray,
                     levels:       tuple      = (0.68, 0.95),
                     loga_grid:    np.ndarray = LOGA_GRID,
                     ) -> dict[float, float]:
    """Empirical coverage at each central credible level.

    For level L, returns the fraction of stars whose true log_age_myr falls
    inside the central L credible interval [p_lo, p_hi] of their posterior,
    where lo = (1 - L) / 2 and hi = 1 - (1 - L) / 2. Well-calibrated → ≈ L.
    """
    cdf = np.cumsum(posteriors * np.gradient(loga_grid), axis=1)
    cdf /= cdf[:, -1:]

    out: dict[float, float] = {}
    for L in levels:
        lo_q, hi_q = (1 - L) / 2, 1 - (1 - L) / 2
        p_lo = np.array([np.interp(lo_q, cdf[i], loga_grid) for i in range(len(posteriors))])
        p_hi = np.array([np.interp(hi_q, cdf[i], loga_grid) for i in range(len(posteriors))])
        out[L] = float(((log_age_true >= p_lo) & (log_age_true <= p_hi)).mean())
    return out
