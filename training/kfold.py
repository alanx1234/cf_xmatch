import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

from .flow    import create_flow
from .data    import make_folds, make_tensors, make_age_sigma
from .train   import train_fold, compute_test_ll
from .infer   import batch_posteriors
from .metrics import compute_residuals
from .constants import PRIOR_LOGPROT


def run_kfold(
    df:              pd.DataFrame,
    test_df:         pd.DataFrame,
    obs_col:         str,
    cond_cols:       list[str],
    n_folds:         int        = 5,
    total_steps:     int        = 5000,
    batch_size:      int        = 256,
    n_samples:       int        = 10,
    use_onecycle:    bool       = False,
    hidden_features: tuple[int] = (64, 64),
    prior_bounds:    tuple      = PRIOR_LOGPROT,
) -> tuple[pd.DataFrame, list, list[StandardScaler],
           list[list[float]], list[list[float]], list[float], np.ndarray]:
    """Orchestrates k-fold: split → train → infer → collect residuals + test LL.

    df       : 90% train/val pool (already split from full dataset).
    test_df  : fixed 10% holdout, never seen during training.

    Returns
    -------
    results_df       : val-fold residuals for all stars in df
    fold_flows       : trained flow per fold
    fold_scalers     : fitted StandardScaler per fold
    train_curves     : per-epoch train loss per fold
    val_curves       : per-epoch val loss per fold
    test_lls         : avg log likelihood on test_df per fold
    posteriors_all   : shape (N, len(LOGA_GRID)), held-out posteriors
    """
    n_cond = len(cond_cols)
    folds  = make_folds(df, n_folds=n_folds)


    all_results    = []
    all_posteriors = []
    fold_flows     = []
    fold_scalers   = []
    train_curves   = []
    val_curves     = []
    test_lls       = []

    for fold_i, (train_df, val_df) in enumerate(folds):
        print(f'\n=== Fold {fold_i + 1} / {n_folds} ===')

        # Scaler fitted on unperturbed train_df — age sampling uses raw log_age later
        x_train, c_train, scaler = make_tensors(train_df, obs_col, cond_cols)
        x_val,   c_val,   _      = make_tensors(val_df,   obs_col, cond_cols, scaler)
        x_test,  c_test,  _      = make_tensors(test_df,  obs_col, cond_cols, scaler)

        log_age_tr,  sigma_lo_tr,  sigma_hi_tr,  has_err_tr  = make_age_sigma(train_df)
        log_age_val, sigma_lo_val, sigma_hi_val, has_err_val = make_age_sigma(val_df)

        torch.manual_seed(42 + fold_i)
        flow = create_flow(n_cond=n_cond, hidden_features=hidden_features)

        tr_curve, vl_curve = train_fold(
            flow         = flow,
            x_train      = x_train,
            c_train      = c_train,
            log_age_tr   = log_age_tr,
            sigma_lo_tr  = sigma_lo_tr,
            sigma_hi_tr  = sigma_hi_tr,
            has_err_tr   = has_err_tr,
            x_val        = x_val,
            c_val        = c_val,
            log_age_val  = log_age_val,
            sigma_lo_val = sigma_lo_val,
            sigma_hi_val = sigma_hi_val,
            has_err_val  = has_err_val,
            scaler       = scaler,
            total_steps  = total_steps,
            batch_size   = batch_size,
            n_samples    = n_samples,
            use_onecycle = use_onecycle,
            prior_bounds = prior_bounds,
        )

        test_ll = compute_test_ll(flow, x_test, c_test)
        print(f'  test avg log-likelihood: {test_ll:.6f}')

        posteriors = batch_posteriors(flow, val_df, obs_col, cond_cols, scaler)
        val_result = compute_residuals(val_df, posteriors)
        val_result['fold'] = fold_i

        all_results.append(val_result)
        all_posteriors.append(posteriors)
        fold_flows.append(flow)
        fold_scalers.append(scaler)
        train_curves.append(tr_curve)
        val_curves.append(vl_curve)
        test_lls.append(test_ll)

    results_df     = pd.concat(all_results, ignore_index=True)
    posteriors_all = np.vstack(all_posteriors)

    mean_test_ll = float(np.mean(test_lls))
    print(f'\n=== Mean test avg log-likelihood across folds: {mean_test_ll:.6f} ===')

    return (results_df, fold_flows, fold_scalers,
            train_curves, val_curves, test_lls, posteriors_all)
