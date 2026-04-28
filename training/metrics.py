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
