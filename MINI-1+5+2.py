# %%
from pathlib import Path
import numpy as np

import src.gdmtools as gdmtools

from cobaya.run import run
from cobaya.log import LoggedError

# %% Define project name and output directory

PROJECT_NAME = "gdm_alpha_5w_2c_fixed_ends"
PROJECT_DIR = Path("/opt/project/") / PROJECT_NAME / ""
#PROJECT_DIR = Path.cwd() / PROJECT_NAME

CHAIN_DIR = PROJECT_DIR / "chains/"
(CHAIN_DIR / "").mkdir(exist_ok=True, parents=True)
CLASS_PATH = "/home/mmeiers/Projects/gdm_cosmology/code/my-class"

COBAYA_PACKAGES_PATH = Path("/software/cobaya_packages")
# %% laod gdm model

gdm_model = gdmtools.yaml.load(PROJECT_DIR / f"{PROJECT_NAME}+model.yaml")

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
    "Omega_gdm_max": None,
    "z_gdm_max": None,
}

# %% setup gdm_likelihood, it has a trivial likelihood but Omega_gdm_max and z_gdm_max are computed here


def gdm_likelihood(_self):
    bg = _self.provider.get_CLASS_background()
    return (
        0,
        {
            "Omega_gdm_max": 0,
            "z_gdm_max": 0,
        },
    )


# %%
gdm_fixed = gdm_model.fixed_settings
gdm_fixed.pop("gdm_z_alpha",None)
gdm_fixed.pop("nap",None)



cobaya_info = dict(
    theory={
        "classy": {
            "extra_args": {
                **gdm_fixed,
                "non_linear": "hmcode",
                "l_max_scalars": 5000,
            },
            # "path": str(CLASS_PATH),
            #'provide': {'get_CLASS_background': None},
        }
    },
    params={**lcdm_params, **gdm_model.params,"gdm_z_alpha":3000},
    likelihood={
        "planck_2018_lowl.TT": None,
        "planck_2018_lowl.EE": None,
        "planck_2018_highl_plik.TTTEEE_lite": None,
        "planck_2018_lensing.clik": None,
        "gdm": {
            "external": gdm_likelihood,
            "output_params": ["Omega_gdm_max", "z_gdm_max"],
            "requires": {"CLASS_background": None},
        },

    },
    sampler=dict(
        minimize=None
    ),
    output=str(CHAIN_DIR / PROJECT_NAME),
    resume= True,
    packages_path=str(COBAYA_PACKAGES_PATH),
)

# %%
upd_info, mcmc = run(cobaya_info, force=True)

