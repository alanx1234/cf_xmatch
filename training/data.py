import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from torch import Tensor


def load_training(path: str, drop_nan_age_err: bool = False) -> pd.DataFrame:
    """Load training_stars.csv and compute all derived log columns.

    drop_nan_age_err=True drops the 273 stars with no age uncertainty (for 1B/1C).
    """
    df = pd.read_csv(path)

    df['log_prot']        = np.log10(df['prot_days'])
    df['log_age_myr']     = np.log10(df['age_gyr'] * 1000)
    df['log_mass_msun']   = np.log10(df['mass_msun'])
    df['log_tau_ce']      = np.log10(df['tau_ce_days'])
    df['log_tau_ce_err']  = np.log10(df['tau_ce_err_days'])
    df['log_mass_err_lo'] = np.log10(df['mass_msun_err_lo'])
    df['log_mass_err_hi'] = np.log10(df['mass_msun_err_hi'])
    df['log_rossby']      = df['log_prot'] - df['log_tau_ce']

    # Only defined for stars with age uncertainties
    has_age_err = df['age_err_lo_gyr'].notna() & df['age_err_hi_gyr'].notna()
    df.loc[has_age_err, 'log_age_err_lo'] = np.log10(df.loc[has_age_err, 'age_err_lo_gyr'] * 1000)
    df.loc[has_age_err, 'log_age_err_hi'] = np.log10(df.loc[has_age_err, 'age_err_hi_gyr'] * 1000)

    # cluster_name: maps each star to its actual cluster for heatmap grouping.
    # - 'multi': star_name holds the cluster name (Hyades, Pleiades, etc.)
    # - 'Mamonova2025': spans 9 distinct clusters — split by age in Myr
    # - field-star compilations: excluded (None) — not single-age populations
    _field_sources = {'MOCADB', 'Engle2023', 'LWRD', 'Pass2022'}
    age_myr = df['age_gyr'] * 1000
    df['cluster_name'] = np.where(
        df['source_paper'] == 'multi', df['star_name'],
        np.where(df['source_paper'] == 'Mamonova2025',
                 'Mamonova2025 (' + age_myr.round(1).astype(str) + ' Myr)',
        np.where(df['source_paper'].isin(_field_sources), None,
                 df['source_paper']))
    )

    if drop_nan_age_err:
        df = df[has_age_err].reset_index(drop=True)

    return df


def make_folds(df: pd.DataFrame,
               n_folds: int = 5) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    """Stratified k-fold on combined age x mass x prot quantile bins (27 strata).

    Ensures each fold has balanced coverage across age, mass, and rotation period,
    as specified by Phil. All 27 strata have >= 5 stars in both the full and
    age-error-filtered datasets.
    """
    age_bin  = pd.qcut(df['log_age_myr'],   q=3, labels=False, duplicates='drop')
    mass_bin = pd.qcut(df['log_mass_msun'], q=3, labels=False, duplicates='drop')
    prot_bin = pd.qcut(df['log_prot'],      q=3, labels=False, duplicates='drop')
    stratum  = age_bin.astype(str) + mass_bin.astype(str) + prot_bin.astype(str)

    skf   = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    folds = []
    for train_idx, val_idx in skf.split(df, stratum):
        folds.append((df.iloc[train_idx].reset_index(drop=True),
                      df.iloc[val_idx].reset_index(drop=True)))
    return folds


def make_tensors(df: pd.DataFrame,
                 obs_col: str,
                 cond_cols: list[str],
                 scaler: StandardScaler | None = None
                 ) -> tuple[Tensor, Tensor, StandardScaler]:
    """Normalize cond_cols to zero mean unit std. Returns (x, c, fitted_scaler).

    Pass the fitted train-fold scaler when transforming the val fold to prevent
    data leakage — the scaler must never see val-fold data during fitting.
    """
    x_vals = df[obs_col].values.astype(np.float32).reshape(-1, 1)
    c_vals = df[cond_cols].values.astype(np.float32)

    if scaler is None:
        scaler = StandardScaler()
        c_norm = scaler.fit_transform(c_vals)
    else:
        c_norm = scaler.transform(c_vals)

    x = torch.tensor(x_vals)
    c = torch.tensor(c_norm)
    return x, c, scaler


def make_age_weights(df: pd.DataFrame,
                     err_lo_col: str = 'age_err_lo_gyr',
                     err_hi_col: str  = 'age_err_hi_gyr') -> Tensor:
    """Normalized inverse age uncertainty weights for 1B loss.

    Weights by 1/(err_lo + err_hi) in Gyr, which naturally upweights young
    clusters whose absolute age errors are smallest. Normalized so mean weight = 1,
    preserving effective learning rate.
    """
    w = 1.0 / (df[err_lo_col].values + df[err_hi_col].values)
    w = w / w.mean()
    return torch.tensor(w.astype(np.float32))


def sample_log_age(df: pd.DataFrame,
                   age_col: str     = 'log_age_myr',
                   err_lo_col: str  = 'age_err_lo_gyr',
                   err_hi_col: str  = 'age_err_hi_gyr') -> pd.DataFrame:
    """Perturb log_age using asymmetric Gaussian sampling in original space.

    Must be called BEFORE make_tensors so sampling occurs before normalization.
    Draws e ~ N(0,1); applies σ_lo if e < 0, σ_hi if e >= 0, where σ is
    converted from Gyr uncertainty into log10(Myr) space.
    """
    df = df.copy()
    age_myr    = df['age_gyr'] * 1000
    err_lo_myr = df[err_lo_col] * 1000
    err_hi_myr = df[err_hi_col] * 1000

    # Convert asymmetric uncertainties to log10(Myr) space.
    # Clip lower bound to 1 Myr to avoid log10 of zero/negative for stars
    # whose err_lo is comparable to or larger than their age.
    age_lo = np.maximum(age_myr - err_lo_myr, 1.0)
    sigma_lo = np.log10(age_myr) - np.log10(age_lo)
    sigma_hi = np.log10(age_myr + err_hi_myr) - np.log10(age_myr)

    eps       = np.random.randn(len(df))
    log_age_s = df[age_col].values.copy()
    log_age_s = np.where(eps < 0,
                         log_age_s + eps * sigma_lo,
                         log_age_s + eps * sigma_hi)

    # Clip to prior bounds to avoid out-of-range samples
    log_age_s       = np.clip(log_age_s, 0, 4.14)
    df[age_col]     = log_age_s
    return df
