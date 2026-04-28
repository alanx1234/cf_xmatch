import numpy as np

# Age grid for posterior evaluation — log10(Myr)
LOGA_GRID = np.linspace(0, 4.14, 1000)

# Prior bounds on log10(P_rot / days) — used in loss outlier component
PRIOR_LOGPROT = (-1.75, 2.5)

# Prior bounds on log10(age / Myr) — used in inference grid
PRIOR_LOGA_MYR = (0, 4.14)

# Outlier probability — fraction of stars not following rotational evolution
# (binaries, misclassified non-members, etc.). Confirm with Phil.
P_OUT = 0.10
W_F   = 1 - P_OUT  # flow weight in loss
