import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import zuko

from .constants import LOGA_GRID
from .plots import (plot_loss, plot_residuals, plot_precision_grid,
                    plot_accuracy_grid, plot_posteriors, plot_prot_space)


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


def kfold_report(results_df: pd.DataFrame, x_col: str) -> None:
    """Full diagnostic panel after k-fold:
    1. Residuals vs x_col
    2. Residuals vs inferred age
    3. Precision grid (x_col vs inferred age)
    4. Residual histogram with median marked.
    """
    plot_residuals(results_df, x_col=x_col)
    plot_residuals(results_df, x_col='inferred_age_gyr')
    plot_precision_grid(results_df, x_col=x_col)
    plot_accuracy_grid(results_df, x_col=x_col)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(results_df['residual_dex'], bins=50,
            color='steelblue', alpha=0.8, edgecolor='none')
    med = results_df['residual_dex'].median()
    ax.axvline(0,   color='black',    lw=1.5)
    ax.axvline(med, color='deeppink', lw=1.5, ls='--',
               label=f'median = {med:.2f} dex')
    ax.set_xlabel('Residual (dex)')
    ax.set_ylabel('N')
    ax.set_title('Residual Distribution  (p50 − true log age)\n'
                 'ideal: centered at 0 with tight spread',
                 fontsize=9)
    ax.legend()
    plt.tight_layout()
    plt.show()


def posterior_report(posteriors: np.ndarray,
                     loga_grid:  np.ndarray,
                     df:         pd.DataFrame) -> None:
    """12 sample posteriors drawn evenly across the age range."""
    plot_posteriors(posteriors, loga_grid, df, n_sample=12)


def prot_space_report(flows:     list[zuko.flows.NSF],
                      df:        pd.DataFrame,
                      cond_cols: list[str],
                      scalers:   list[StandardScaler],
                      loga_grid: np.ndarray = LOGA_GRID) -> None:
    """Figure 7 equivalent averaged across all fold models."""
    plot_prot_space(flows, df, cond_cols, scalers, loga_grid)
