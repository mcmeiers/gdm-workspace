#!/usr/bin/env python
# coding: utf-8

# import file/os packages
import sys
from git import Repo
from pathlib import Path
from ruamel.yaml import YAML

# import math packages
import numpy as np

# import cobaya packages
from cobaya.run import run
from cobaya.log import LoggedError


PROJECT_DIR = Path(Repo(".", search_parent_directories=True).working_tree_dir)
sys.path.append(str(PROJECT_DIR))

import src.gdmtools as gdm

from mpi4py import MPI

yaml = YAML(typ="unsafe")


with open(PROJECT_DIR / "system.config") as f:
    local_config = yaml.load(f)

OUTPUT_DIR = PROJECT_DIR / local_config.get("output-dir", "output")
MODEL_DIR = PROJECT_DIR / local_config.get("models-dir", "models")

CLASS_PATH = PROJECT_DIR / local_config.get("class-dir", "../code/my-class/")
COBAYA_PACKAGES_PATH = PROJECT_DIR / local_config.get("cobaya-pkg-dir", "../")


MODEL_NAME = "ede-1+5+2"
gdm_fluid = gdm.yaml.load(MODEL_DIR / f"{MODEL_NAME}.gdm.yaml")
lcdm_params = yaml.load(MODEL_DIR / "lcdm_params_v0.yaml")

redshift_samples = np.logspace(0, 6, 13)
H_sample_params = {f"H_{z_idx}": None for z_idx in range(len(redshift_samples))}


def gdm_derived_params(_self):
    bg = _self.provider.get_CLASS_background()
    Omega_gdm = bg["(.)rho_gdm"] / bg["H [1/Mpc]"] ** 2
    Omega_gdm_max_idx = np.argmax(Omega_gdm)
    H0s = dict(zip(H_sample_params.keys(), _self.provider.get_Hubble(redshift_samples)))
    return (
        0,
        {
            "Omega_gdm_max": Omega_gdm[Omega_gdm_max_idx],
            "z_gdm_max": bg["z"][Omega_gdm_max_idx],
        }
        | H0s,
    )


import contextlib

derived_params = {
    "Omega_gdm_max": None,
    "z_gdm_max": None,
    **H_sample_params,
}


cobaya_info = dict(
    theory={
        "classy": {
            "extra_args": {
                **gdm_fluid.cobaya_fixed_settings,
                "non_linear": "hmcode",
                "l_max_scalars": 5000,
            },
            "path": str(CLASS_PATH),
        }
    },
    params={**lcdm_params, **gdm_fluid.params, **derived_params},
    likelihood={
        "planck_2018_lowl.TT": None,
        "planck_2018_lowl.EE": None,
        "planck_2018_highl_plik.TTTEEE_lite": None,
        "planck_2018_lensing.clik": None,
        "gdm_derived": {
            "external": gdm_derived_params,
            "output_params": ["Omega_gdm_max", "z_gdm_max"]
            + list(H_sample_params.keys()),
            "requires": {"CLASS_background": None, "Hubble": {"z": redshift_samples}},
        },
    },
    sampler=dict(
        polychord={
            "oversample_power": 0.4,
        }
    ),
    output=str(OUTPUT_DIR / MODEL_NAME),
    resume=True,
)


comm = MPI.COMM_WORLD
rank = comm.Get_rank()

success = False

with contextlib.suppress(LoggedError):
    upd_info, mcmc = run(cobaya_info, resume=True)
    success = True
# Did it work? (e.g. did not get stuck)
success = all(comm.allgather(success))

if not success and rank == 0:
    print("Sampling failed!")


#
