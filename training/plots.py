import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from sklearn.preprocessing import StandardScaler
import zuko

from .constants import LOGA_GRID, PRIOR_LOGPROT


def plot_loss(loss_curve: list[float]) -> None:
    """Training loss vs step for one fold."""
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(loss_curve, lw=1, color='steelblue')
    ax.set_xlabel('Step')
    ax.set_ylabel('Loss')
    ax.set_title('Training Loss')
    plt.tight_layout()
    plt.show()


def plot_residuals(df: pd.DataFrame, x_col: str) -> None:
    """Residuals (p50 - true log_age_myr) vs x_col.

    x_col: 'log_mass_msun', 'log_tau_ce', or 'inferred_age_gyr'.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(
        df[x_col], df['residual_dex'],
        yerr=[df['p50'] - df['p16'], df['p84'] - df['p50']],
        fmt='none', alpha=0.3, color='steelblue', lw=0.5,
    )
    ax.scatter(df[x_col], df['residual_dex'], s=3, alpha=0.4, color='steelblue')
    ax.axhline(0, color='black', lw=1)
    ax.axhline(df['residual_dex'].median(), color='deeppink',
               lw=1, ls='--', label=f'median = {df["residual_dex"].median():.2f} dex')
    ax.set_xlabel(x_col)
    ax.set_ylabel('Residual (dex)')
    ax.legend()
    plt.tight_layout()
    plt.show()


def plot_precision_grid(df: pd.DataFrame, x_col: str) -> None:
    """2D heatmap of median p84-p16 precision binned by x_col and inferred age.

    Equivalent to ChronoFlow figures 15/16.
    """
    x_bins   = np.linspace(df[x_col].min(), df[x_col].max(), 11)
    age_bins = np.linspace(df['p50'].min(), df['p50'].max(), 11)

    grid = np.full((len(age_bins) - 1, len(x_bins) - 1), np.nan)
    for i in range(len(age_bins) - 1):
        for j in range(len(x_bins) - 1):
            mask = (
                (df['p50']   >= age_bins[i]) & (df['p50']   < age_bins[i + 1]) &
                (df[x_col]   >= x_bins[j])   & (df[x_col]   < x_bins[j + 1])
            )
            if mask.sum() >= 3:
                grid[i, j] = (df.loc[mask, 'p84'] - df.loc[mask, 'p16']).median()

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(grid, origin='lower', aspect='auto',
                   extent=[x_bins[0], x_bins[-1], age_bins[0], age_bins[-1]],
                   cmap='viridis_r')
    plt.colorbar(im, ax=ax, label='Median precision (dex)')
    ax.set_xlabel(x_col)
    ax.set_ylabel('Inferred log age (Myr)')
    ax.set_title('Posterior Precision Grid')
    plt.tight_layout()
    plt.show()


def plot_posteriors(posteriors: np.ndarray,
                    loga_grid:  np.ndarray,
                    df:         pd.DataFrame,
                    n_sample:   int = 12) -> None:
    """n_sample posteriors drawn evenly across the age range, true age marked."""
    sorted_idx = np.argsort(df['log_age_myr'].values)[
        np.linspace(0, len(df) - 1, n_sample, dtype=int)
    ]

    ncols = 4
    nrows = int(np.ceil(n_sample / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, nrows * 3))
    axes = axes.flatten()

    for k, idx in enumerate(sorted_idx):
        ax = axes[k]
        ax.plot(loga_grid, posteriors[idx], color='steelblue', lw=1.5)
        ax.axvline(df['log_age_myr'].iloc[idx], color='deeppink',
                   lw=1.5, ls='--', label='true age')
        ax.set_xlabel('log age (Myr)')
        ax.set_ylabel('p(age)')
        ax.legend(fontsize=7)

    for ax in axes[n_sample:]:
        ax.set_visible(False)

    plt.suptitle('Sample Age Posteriors', y=1.02)
    plt.tight_layout()
    plt.show()


def plot_prot_space(flows:     list[zuko.flows.NSF],
                    df:        pd.DataFrame,
                    cond_cols: list[str],
                    scalers:   list[StandardScaler],
                    loga_grid: np.ndarray = LOGA_GRID) -> None:
    """P(P_rot | age, feature) heatmap per cluster overlaid with observations.

    Averages log-probs across all fold flows before plotting.
    Equivalent to ChronoFlow figure 7.
    """
    age_col_idx     = cond_cols.index('log_age_myr')
    feature_col_idx = 1  # second conditioning var (log_mass or log_tau_ce)
    feature_col     = cond_cols[feature_col_idx]

    clusters    = df['source_paper'].unique()
    n_clusters  = len(clusters)
    ncols       = 4
    nrows       = int(np.ceil(n_clusters / ncols))
    res         = 80  # grid resolution — lower than ChronoFlow to keep it fast

    logP_grid    = np.linspace(PRIOR_LOGPROT[0], PRIOR_LOGPROT[1], res)
    feature_grid = np.linspace(df[feature_col].min(), df[feature_col].max(), res)

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * 3.5, nrows * 3.5),
                             sharex=True, sharey=True)
    axes = axes.flatten()

    for k, cluster in enumerate(sorted(clusters)):
        ax      = axes[k]
        cl_df   = df[df['source_paper'] == cluster]
        log_age = cl_df['log_age_myr'].median()

        # Build (res*res, n_cond) grid of conditioning vectors
        PP, FF   = np.meshgrid(logP_grid, feature_grid, indexing='ij')
        c_median = df[cond_cols].median().values.astype(np.float32)

        c_grid = np.tile(c_median, (res * res, 1))
        c_grid[:, age_col_idx]     = log_age
        c_grid[:, feature_col_idx] = FF.ravel()

        x_grid = torch.tensor(PP.ravel().astype(np.float32)).unsqueeze(1)

        # Average log-probs across all fold models
        log_probs = np.zeros(res * res, dtype=np.float32)
        for flow, scaler in zip(flows, scalers):
            c_norm = scaler.transform(c_grid).astype(np.float32)
            c_t    = torch.tensor(c_norm)
            with torch.no_grad():
                log_probs += flow(c_t).log_prob(x_grid).numpy()
        log_probs = (log_probs / len(flows)).reshape(res, res)

        ax.imshow(log_probs, origin='lower', aspect='auto',
                  extent=[feature_grid[0], feature_grid[-1],
                          PRIOR_LOGPROT[0], PRIOR_LOGPROT[1]],
                  cmap='bone', vmin=-3, vmax=2)
        ax.scatter(cl_df[feature_col], np.log10(cl_df['prot_days']),
                   s=6, color='deeppink', alpha=0.7, zorder=2)
        age_myr = 10 ** log_age
        ax.set_title(f'{cluster}\n{age_myr:.0f} Myr', fontsize=8)
        ax.set_xlabel(feature_col, fontsize=7)
        ax.set_ylabel('log P_rot (d)', fontsize=7)

    for ax in axes[n_clusters:]:
        ax.set_visible(False)

    plt.suptitle('P(P_rot | age, feature) per cluster', y=1.01)
    plt.tight_layout()
    plt.show()
