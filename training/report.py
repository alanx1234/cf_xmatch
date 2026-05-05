import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import zuko

from .constants import LOGA_GRID
from .plots import (plot_loss, plot_residuals, plot_precision_grid,
                    plot_accuracy_grid, plot_posteriors, plot_prot_space,
                    plot_residual_panels, plot_cluster_residual_densities)


def training_report(loss_curves: list[list[float]]) -> None:
    """All 5 fold loss curves overlaid on one figure."""
    fig, ax = plt.subplots(figsize=(8, 4))
    for i, lc in enumerate(loss_curves):
        ax.plot(lc, lw=1, alpha=0.7, label=f'Fold {i}')
    ax.set_xlabel('Step')
    ax.set_ylabel('Loss')
    ax.set_title('Training Loss — All Folds')
    ax.legend()
    plt.tight_layout()
    plt.show()


def kfold_report(results_df: pd.DataFrame) -> None:
    """Residual panels, per-cluster density distributions, and precision/accuracy grids."""
    plot_residual_panels(results_df)
    plot_cluster_residual_densities(results_df)
    plot_precision_grid(results_df, x_col='mass_msun')
    plot_accuracy_grid(results_df, x_col='mass_msun')


def posterior_report(posteriors: np.ndarray,
                     loga_grid:  np.ndarray,
                     df:         pd.DataFrame) -> None:
    """12 sample posteriors drawn evenly across the age range."""
    plot_posteriors(posteriors, loga_grid, df, n_sample=12)


def coverage_report(results_df: pd.DataFrame,
                    expected: float = 0.68) -> pd.DataFrame:
    """Print nominal 68% posterior coverage and return fold-level coverage."""
    valid = results_df[['log_age_myr', 'p16', 'p84']].notna().all(axis=1)
    df_valid = results_df.loc[valid].copy()
    inside_68 = (
        (df_valid['log_age_myr'] >= df_valid['p16'])
        & (df_valid['log_age_myr'] <= df_valid['p84'])
    )

    n = len(df_valid)
    cover68 = inside_68.mean()
    binom_sigma = np.sqrt(expected * (1 - expected) / n)
    offset_sigma = (cover68 - expected) / binom_sigma

    print('=== 68% posterior coverage check ===')
    print(f'N valid stars       : {n}')
    print(f'Observed coverage   : {cover68:.3f} ({cover68*100:.1f}%)')
    print(f'Expected coverage   : {expected:.3f} +/- {binom_sigma:.3f} (binomial 1-sigma)')
    print(f'Coverage offset     : {offset_sigma:+.2f} sigma')

    if cover68 < expected - binom_sigma:
        print('Interpretation      : under-covering; posteriors are likely too narrow / overconfident.')
    elif cover68 > expected + binom_sigma:
        print('Interpretation      : over-covering; posteriors are likely too broad / conservative.')
    else:
        print('Interpretation      : close to nominal 68% calibration.')

    return (
        df_valid
        .assign(inside_68=inside_68.to_numpy())
        .groupby('fold')['inside_68']
        .agg(cover68='mean', n='size')
    )


def prot_space_report(flows:     list[zuko.flows.NSF],
                      df:        pd.DataFrame,
                      cond_cols: list[str],
                      scalers:   list[StandardScaler],
                      loga_grid: np.ndarray = LOGA_GRID) -> None:
    """Figure 7 equivalent averaged across all fold models."""
    plot_prot_space(flows, df, cond_cols, scalers, loga_grid)
