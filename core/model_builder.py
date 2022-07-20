# %%
import numpy as np

import sys
import git
from pathlib import Path

REPO_DIR = Path(git.Repo(".", search_parent_directories=True).working_tree_dir)
sys.path.append(str(REPO_DIR))

import src.gdmtools as gdm

# %%

DATA_DIR = REPO_DIR / "data"
MODEL_DIR = REPO_DIR / "models"

# %% Change below to set model info

MODEL_NAME = "gdm-step-v2"

free_knots_log10a = tuple(float(x) for x in np.linspace(-4.5, -3.5, 3))
fixed_knots = [(-14.0, 1 / 3.0), (-5.0, 1 / 3.0), (-3, 1 / 3.0), (0.0, 1 / 3.0)]
range_filter = None
# (
#     (-14, {"min": -1, "max": 1 / 3.0}),
#     (-7, {"min": -1, "max": 1}),
#     (-2, {"min": 1 / 3, "max": 1}),
# )


wMdl = gdm.wModel(
    free_knots_log10a, fixed_knots, range_filter=range_filter
)  # range_filter=range_filter)

gdmMdl = gdm.gdmModel(
    w_model=wMdl,
    alpha={"prior": {"min": 0, "max": 0.05}},
    c_eff2=None,
    c_vis2=0,
    z_alpha=3000,
)

with open(MODEL_DIR / f"{MODEL_NAME}.gdm.yaml", "w") as f:
    gdm.yaml.dump(gdmMdl, f)

# %%
