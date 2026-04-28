import zuko


def create_flow(n_cond: int, n_transforms: int = 3) -> zuko.flows.NSF:
    """Returns NSF(1, n_cond, transforms=n_transforms).

    n_cond varies by model variant:
      1A mass: 4  (log_age, log_mass, log_mass_err_lo, log_mass_err_hi)
      1A tauc: 3  (log_age, log_tau_ce, log_tau_ce_err)
      1B mass: 4  (same as 1A — weighting only, no extra conditioning)
      1B tauc: 3  (same as 1A tauc)
      1C mass: 4  (log_age_sampled replaces log_age)
      1C tauc: 3  (log_age_sampled replaces log_age)
    """
    return zuko.flows.NSF(1, n_cond, transforms=n_transforms)
