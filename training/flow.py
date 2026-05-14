import zuko


def create_flow(n_cond:          int,
                n_transforms:    int        = 3,
                hidden_features: tuple[int] = (64, 64)) -> zuko.flows.NSF:
    """Returns NSF(1, n_cond, transforms=n_transforms, hidden_features=hidden_features).

    hidden_features controls the MLP depth/width inside each spline transform.
    Baseline: (64, 64) — 2 hidden layers of 64 units (zuko default).
    2x layers: (64, 64, 64, 64) — 4 hidden layers.
    2x width:  (128, 128) — 2 wider hidden layers.
    """
    return zuko.flows.NSF(1, n_cond,
                          transforms      = n_transforms,
                          hidden_features = hidden_features)
