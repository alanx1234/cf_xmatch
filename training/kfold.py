import pandas as pd
import torch
from typing import Callable
from sklearn.preprocessing import StandardScaler
import zuko

from .flow import create_flow
from .data import make_folds, make_tensors, make_age_weights, sample_log_age
from .train import train_fold
from .infer import batch_posteriors
from .metrics import compute_residuals
from .constants import LOGA_GRID


def run_kfold(df:            pd.DataFrame,
              obs_col:       str,
              cond_cols:     list[str],
              n_folds:       int             = 5,
              steps:         int             = 5000,
              age_weights:   bool            = False,
              age_sample_fn: Callable | None = None,
              age_col:       str | None      = None,
              err_lo_col:    str | None      = None,
              err_hi_col:    str | None      = None,
              ) -> tuple[pd.DataFrame, list[zuko.flows.NSF],
                         list[StandardScaler], list[list[float]]]:
    """Orchestrates k-fold: split → train → infer → collect residuals.

    age_weights=True  : activates 1B inverse-uncertainty loss weighting.
    age_sample_fn     : pass sample_log_age for 1C stochastic age sampling.
    age_col/err_*_col : required when age_sample_fn is provided.

    Returns (results_df, fold_flows, fold_scalers, loss_curves).
    Posteriors are computed separately in notebooks via batch_posteriors
    to keep orchestration out of this function.
    """
    n_cond = len(cond_cols)
    folds  = make_folds(df, n_folds=n_folds)

    all_results  = []
    fold_flows   = []
    fold_scalers = []
    loss_curves  = []

    for fold_i, (train_df, val_df) in enumerate(folds):
        print(f'\n=== Fold {fold_i + 1} / {n_folds} ===')

        # 1C: sample log_age BEFORE normalization
        if age_sample_fn is not None:
            train_df_s = age_sample_fn(train_df, age_col, err_lo_col, err_hi_col)
        else:
            train_df_s = train_df

        x_train, c_train, scaler = make_tensors(train_df_s, obs_col, cond_cols)
        x_val,   c_val,   _      = make_tensors(val_df,     obs_col, cond_cols, scaler)

        w_age = make_age_weights(train_df) if age_weights else None

        flow = create_flow(n_cond=n_cond)
        lc   = train_fold(
            flow          = flow,
            x             = x_train,
            c             = c_train,
            steps         = steps,
            age_weights   = w_age,
            age_sample_fn = age_sample_fn,
            age_df        = train_df if age_sample_fn else None,
            age_col       = age_col,
            err_lo_col    = err_lo_col,
            err_hi_col    = err_hi_col,
            cond_cols     = cond_cols,
            scaler        = scaler,
        )

        # Infer on val fold using this fold's model
        posteriors = batch_posteriors(flow, val_df, obs_col, cond_cols, scaler)
        val_result = compute_residuals(val_df, posteriors)
        val_result['fold'] = fold_i

        all_results.append(val_result)
        fold_flows.append(flow)
        fold_scalers.append(scaler)
        loss_curves.append(lc)

    results_df = pd.concat(all_results, ignore_index=True)
    return results_df, fold_flows, fold_scalers, loss_curves
