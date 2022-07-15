# %%
from pathlib import Path

import numpy as np

from cobaya.run import run

# %% Define project name and output directory
PROJECT_NAME = "debug_minimizer"
import os

PROJECT_DIR = Path(os.getcwd()) / PROJECT_NAME

CHAIN_DIR = PROJECT_DIR / "chains/"
(CHAIN_DIR / "").mkdir(exist_ok=True, parents=True)

# COBAYA_PACKAGES_PATH = Path("/software/cobaya_packages")
CLASS_PATH = "/home/mcmeiers/Projects/gdm_cosmology/code/my_class"  # COBAYA_PACKAGES_PATH / "code/class_public-designer/"


# %% create wModel

log10a_knots = np.array([-14.0, -4.5, -4.0, -3.5, -3.0, -2.5, 0.0])

w_vals_fn = 'lambda w_0, w_1, w_2, w_3, w_4: "-1.0," + ",".join(map(str, [w_0, w_1, w_2, w_3, w_4])) + ",1.0"'

gdm_params = {
    "gdm_alpha": {"prior": {"max": 0.1, "min": 0}, "latex": "\\alpha_{gdm}"},
    "w_0": {"prior": {"max": 1, "min": -1}, "drop": True},
    "w_1": {"prior": {"max": 1, "min": -1}, "drop": True},
    "w_2": {"prior": {"max": 1, "min": -1}, "drop": True},
    "w_3": {"prior": {"max": 1, "min": -1}, "drop": True},
    "w_4": {"prior": {"max": 1, "min": -1}, "drop": True},
    "gdm_w_vals": {"value": w_vals_fn, "derived": False},
    "gdm_c_eff2": {"prior": {"max": 1, "min": 0}},
    "gdm_c_vis2": {"prior": {"max": 1, "min": 0}},
}

gdm_fixed = {
    "gdm_log10a_vals": "-14,-4.5,-4.0,-3.5,-3.0,-2.5,0",
    "gdm_interpolation_order": 1,
    "gdm_z_alpha": 3000,
}

# %% lcdm model params

lcdm_params = {
    "logA": {
        "prior": {"min": 1.61, "max": 3.91},
        "ref": {"dist": "norm", "loc": 3.05, "scale": 0.001},
        "proposal": 0.001,
        "latex": "\\log(10^{10} A_\\mathrm{s})",
        "drop": True,
    },
    "A_s": {"value": "lambda logA: 1e-10*np.exp(logA)", "latex": "A_\\mathrm{s}"},
    "n_s": {
        "prior": {"min": 0.8, "max": 1.2},
        "ref": {"dist": "norm", "loc": 0.965, "scale": 0.004},
        "proposal": 0.002,
        "latex": "n_\\mathrm{s}",
    },
    "theta_s_1e2": {
        "prior": {"min": 0.5, "max": 10},
        "ref": {"dist": "norm", "loc": 1.0416, "scale": 0.0004},
        "proposal": 0.0002,
        "latex": "100\\theta_\\mathrm{s}",
        "drop": True,
    },
    "100*theta_s": {"value": "lambda theta_s_1e2: theta_s_1e2", "derived": False},
    "H0": {"min": 60, "max": 80, "latex": "H_0"},
    "sigma8": {"latex": "\sigma_8"},
    "omega_b": {
        "prior": {"min": 0.005, "max": 0.1},
        "ref": {"dist": "norm", "loc": 0.0224, "scale": 0.0001},
        "proposal": 0.0001,
        "latex": "\\Omega_\\mathrm{b} h^2",
    },
    "omega_cdm": {
        "prior": {"min": 0.001, "max": 0.99},
        "ref": {"dist": "norm", "loc": 0.12, "scale": 0.001},
        "proposal": 0.0005,
        "latex": "\\Omega_\\mathrm{c} h^2",
    },
    "z_reio": {
        "prior": {"min": 6, "max": 9},
        "ref": {"dist": "norm", "loc": 7.5, "scale": 0.75},
        "proposal": 0.4,
    },
    "tau_reio": {"latex": "\\tau_\\mathrm{reio}"},
}

# %%


cobaya_info = dict(
    theory={
        "classy": {
            "extra_args": {
                **gdm_fixed,
                "non_linear": "hmcode",
            },
            "path": str(CLASS_PATH),
        }
    },
    params={**lcdm_params, **gdm_params},
    likelihood={
        "planck_2018_lowl.TT": None,
        "planck_2018_lowl.EE": None,
        "planck_2018_highl_plik.TTTEEE_lite": None,
        "planck_2018_lensing.clik": None,
    },
    sampler={"minimize": None},
    output=PROJECT_DIR / PROJECT_NAME,
)

cobaya_info["params"].update({"gdm_alpha": 0.05})

# %%
updated_info, sampler = run(cobaya_info, force=True, debug=True)
# %%
updated_info
# %%
from classy import Class

cosmo = Class()
cosmo.set(
    {
        "A_s": 2.109843654335196e-09,
        "n_s": 0.9682256778657358,
        "100*theta_s": 1.0416570151018314,
        "omega_b": 0.022344649211073528,
        "omega_cdm": 0.12026790581268844,
        "z_reio": 6.566572924175148,
        "gdm_alpha": 0.05,
        "gdm_w_vals": "-1.0,0.4160833037723448,0.7642675107047947,0.5682936521397142,0.36458195536282445,0.23224825743549893,1.0",
        "gdm_c_eff2": 0.4883049724964518,
        "gdm_c_vis2": 0.40073061104874297,
        "gdm_log10a_vals": "-14,-4.5,-4.0,-3.5,-3.0,-2.5,0",
        "gdm_interpolation_order": 1,
        "gdm_z_alpha": 3000,
        "non_linear": "hmcode",
        "output": "tCl lCl mPk pCl",
        "l_max_scalars": 2508,
        "lensing": "yes",
        "P_k_max_1/Mpc": 1,
    }
)
cosmo.compute()
# %%
