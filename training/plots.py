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
    ax.set_title(f'Age Residuals vs {x_col}\n'
                 r'residual = p50 $-$ true log age   |   error bars = p16–p84   |   closer to 0 = better',
                 fontsize=9)
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
    ax.set_title('Posterior Precision Grid  (p84 − p16)\n'
                 'yellow = tight/confident   |   purple = wide/uncertain',
                 fontsize=9)
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
        ax   = axes[k]
        row  = df.iloc[idx]
        mass = row['mass_msun'] if 'mass_msun' in df.columns else None
        ax.plot(loga_grid, posteriors[idx], color='steelblue', lw=1.5)
        ax.axvline(row['log_age_myr'], color='deeppink',
                   lw=1.5, ls='--', label='true age')
        title = f"{row['source_paper']}\n{10**row['log_age_myr']:.0f} Myr"
        if mass is not None:
            title += f"  |  {mass:.2f} M☉"
        ax.set_title(title, fontsize=7)
        ax.set_xlabel('log age (Myr)', fontsize=7)
        ax.set_ylabel('p(age)', fontsize=7)
        ax.legend(fontsize=6)

    for ax in axes[n_sample:]:
        ax.set_visible(False)

    plt.suptitle('Sample Age Posteriors', y=1.02)
    plt.tight_layout()
    plt.show()


def plot_accuracy_grid(df: pd.DataFrame, x_col: str) -> None:
    """2D heatmap of median residual_dex binned by x_col and inferred age.

    Equivalent to ChronoFlow figure 18. Shows where the model is systematically
    biased — blue = underestimates age, red = overestimates age.
    """
    x_bins   = np.linspace(df[x_col].min(), df[x_col].max(), 11)
    age_bins = np.linspace(df['p50'].min(), df['p50'].max(), 11)

    grid = np.full((len(age_bins) - 1, len(x_bins) - 1), np.nan)
    for i in range(len(age_bins) - 1):
        for j in range(len(x_bins) - 1):
            mask = (
                (df['p50']  >= age_bins[i]) & (df['p50']  < age_bins[i + 1]) &
                (df[x_col]  >= x_bins[j])   & (df[x_col]  < x_bins[j + 1])
            )
            if mask.sum() >= 3:
                grid[i, j] = df.loc[mask, 'residual_dex'].median()

    abs_max = np.nanmax(np.abs(grid))
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(grid, origin='lower', aspect='auto',
                   extent=[x_bins[0], x_bins[-1], age_bins[0], age_bins[-1]],
                   cmap='RdBu', vmin=-abs_max, vmax=abs_max)
    plt.colorbar(im, ax=ax, label='Median residual (dex)')
    ax.set_xlabel(x_col)
    ax.set_ylabel('Inferred log age (Myr)')
    ax.set_title('Accuracy Grid  (median residual per bin)\n'
                 'blue = model overestimates age   |   red = underestimates   |   white = unbiased',
                 fontsize=9)
    plt.tight_layout()
    plt.show()


def plot_posteriors_by_age(posteriors: np.ndarray,
                           loga_grid:  np.ndarray,
                           df:         pd.DataFrame,
                           n_rows:     int = 4,
                           n_cols:     int = 4) -> None:
    """n_rows x n_cols posteriors organised by age range, true age marked.

    Rows run youngest (top) to oldest (bottom). Within each bin, cluster stars
    (non-field sources) are preferred; field compilations fill gaps only.
    True age (pink dashed) and predicted p50 (orange dotted) both shown.
    """
    _field_sources = {'MOCADB', 'Engle2023', 'LWRD', 'Pass2022'}
    age_vals = df['log_age_myr'].values
    age_bins = np.linspace(age_vals.min(), age_vals.max(), n_rows + 1)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, n_rows * 3),
                              sharey=False)

    for row_i in range(n_rows):
        lo, hi = age_bins[row_i], age_bins[row_i + 1]
        in_bin = (age_vals >= lo) & (age_vals <= hi if row_i == n_rows - 1 else age_vals < hi)
        idx_in_bin = np.where(in_bin)[0]

        # Prefer cluster stars within the bin
        if 'source_paper' in df.columns:
            is_field = df['source_paper'].isin(_field_sources).values
            cluster_idx = idx_in_bin[~is_field[idx_in_bin]]
            pool = cluster_idx if len(cluster_idx) >= n_cols else idx_in_bin
        else:
            pool = idx_in_bin

        if len(pool) >= n_cols:
            sorted_pool = pool[np.argsort(age_vals[pool])]
            pick = sorted_pool[np.linspace(0, len(sorted_pool) - 1, n_cols, dtype=int)]
        else:
            pick = pool

        for col_i, idx in enumerate(pick):
            ax  = axes[row_i, col_i]
            row = df.iloc[idx]
            ax.plot(loga_grid, posteriors[idx], color='steelblue', lw=1.5)
            ax.axvline(row['log_age_myr'], color='deeppink',
                       lw=1.5, ls='--', label='true age')
            if 'p50' in df.columns:
                ax.axvline(row['p50'], color='darkorange',
                           lw=1.5, ls=':', label='p50')
            title = f"{row['source_paper']}\n{10**row['log_age_myr']:.0f} Myr"
            if 'mass_msun' in df.columns:
                title += f"  |  {row['mass_msun']:.2f} M☉"
            if 'prot_days' in df.columns:
                title += f"  |  {row['prot_days']:.1f} d"
            ax.set_title(title, fontsize=7)
            ax.set_xlabel('log age (Myr)', fontsize=7)
            ax.legend(fontsize=6)

        for col_i in range(len(pick), n_cols):
            axes[row_i, col_i].set_visible(False)

        lo_myr = 10 ** lo
        hi_myr = 10 ** hi
        axes[row_i, 0].set_ylabel(
            f'{lo_myr:.0f}–{hi_myr:.0f} Myr\np(age)', fontsize=7)

    plt.suptitle('Sample Age Posteriors by Age Range  (youngest → oldest)',
                 y=1.02)
    plt.tight_layout()
    plt.show()


def plot_prot_space(flows:     list[zuko.flows.NSF],
                    df:        pd.DataFrame,
                    cond_cols: list[str],
                    scalers:   list[StandardScaler],
                    loga_grid: np.ndarray = LOGA_GRID,
                    min_stars: int        = 5,
                    n_rows:    int        = 5,
                    n_cols:    int        = 4) -> None:
    """P(P_rot | age, feature) heatmap in a n_rows x n_cols grid ordered by age.

    Divides the log-age range into n_rows bins; picks the n_cols largest clusters
    per bin so panels run youngest to oldest left-to-right, top-to-bottom.
    Averages log-probs across all fold flows. Equivalent to ChronoFlow figure 7.
    """
    age_col_idx     = cond_cols.index('log_age_myr')
    feature_col_idx = 1
    feature_col     = cond_cols[feature_col_idx]
    cluster_col     = 'cluster_name' if 'cluster_name' in df.columns else 'source_paper'

    counts        = df[cluster_col].dropna().value_counts()
    valid_clusters = counts[counts >= min_stars].index.tolist()
    meta = pd.DataFrame({
        'cluster' : valid_clusters,
        'log_age' : [df.loc[df[cluster_col] == c, 'log_age_myr'].median() for c in valid_clusters],
        'n'       : [counts[c] for c in valid_clusters],
    }).sort_values('log_age').reset_index(drop=True)

    # Pick n_cols largest clusters per age bin
    bins = np.linspace(meta['log_age'].min(), meta['log_age'].max(), n_rows + 1)
    picked = []
    for i in range(n_rows):
        lo, hi  = bins[i], bins[i + 1]
        in_bin  = meta[(meta['log_age'] >= lo) & (meta['log_age'] <= hi if i == n_rows - 1 else meta['log_age'] < hi)]
        picked.append(in_bin.nlargest(n_cols, 'n'))
    selected = pd.concat(picked).reset_index(drop=True)

    res          = 80
    logP_grid    = np.linspace(PRIOR_LOGPROT[0], PRIOR_LOGPROT[1], res)
    feature_grid = np.linspace(df[feature_col].min(), df[feature_col].max(), res)
    c_median     = df[cond_cols].median().values.astype(np.float32)

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(n_cols * 3.5, n_rows * 3.5),
                             sharex=True, sharey=True)
    axes = axes.flatten()

    for k, row in selected.iterrows():
        ax      = axes[k]
        cluster = row['cluster']
        log_age = row['log_age']
        cl_df   = df[df[cluster_col] == cluster]

        PP, FF = np.meshgrid(logP_grid, feature_grid, indexing='ij')
        c_grid = np.tile(c_median, (res * res, 1))
        c_grid[:, age_col_idx]     = log_age
        c_grid[:, feature_col_idx] = FF.ravel()
        x_grid = torch.tensor(PP.ravel().astype(np.float32)).unsqueeze(1)

        log_probs = np.zeros(res * res, dtype=np.float32)
        for flow, scaler in zip(flows, scalers):
            c_norm = scaler.transform(c_grid).astype(np.float32)
            with torch.no_grad():
                log_probs += flow(torch.tensor(c_norm)).log_prob(x_grid).numpy()
        log_probs = (log_probs / len(flows)).reshape(res, res)

        ax.imshow(log_probs, origin='lower', aspect='auto',
                  extent=[feature_grid[0], feature_grid[-1],
                          PRIOR_LOGPROT[0], PRIOR_LOGPROT[1]],
                  cmap='bone', vmin=-3, vmax=2)
        ax.scatter(cl_df[feature_col], np.log10(cl_df['prot_days']),
                   s=6, color='deeppink', alpha=0.7, zorder=2)
        ax.set_title(f'{cluster}\n{10**log_age:.0f} Myr  (N={row["n"]:.0f})', fontsize=8)
        ax.set_xlabel(feature_col, fontsize=7)
        ax.set_ylabel('log P_rot (d)', fontsize=7)

    for ax in axes[len(selected):]:
        ax.set_visible(False)

    plt.suptitle('P(P_rot | age, feature) per cluster  —  youngest → oldest', y=1.01)
    plt.tight_layout()
    plt.show()
