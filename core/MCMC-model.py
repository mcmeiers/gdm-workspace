# %%
import os
import sys
import git
import click
from pathlib import Path

REPO_DIR = Path(git.Repo(".", search_parent_directories=True).working_tree_dir)
sys.path.append(str(REPO_DIR))

import numpy as np

import src.gdmtools as gdm

from cobaya.run import run
from cobaya.log import LoggedError


from mpi4py import MPI

# %%


def main(path_to_model):
    MODEL_NAME = os.path.split(path_to_model)[1].split(".")[0]
    OUTPUT_DIR = Path("/opt/project/output") / MODEL_NAME
    (OUTPUT_DIR / "").mkdir(exist_ok=True, parents=True)

    CHAIN_DIR = OUTPUT_DIR / "chains/"
    (CHAIN_DIR / "").mkdir(exist_ok=True, parents=True)
    CHAIN_PATH = CHAIN_DIR / MODEL_NAME

    COBAYA_PACKAGES_PATH = Path("/software/cobaya_packages")
    CLASS_PATH = COBAYA_PACKAGES_PATH / "code/class_public-designer/"

    gdm_model = gdm.yaml.load(path_to_model)

    redshift_samples = np.logspace(0, 6, 13)

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
            "prior": {"min": 0.9, "max": 1.15},
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
        "tau_reio": {
            "prior": {"min": 0.04, "max": 0.07},
            "ref": {"dist": "norm", "loc": 0.0543, "scale": 0.0073},
            "proposal": 0.0073,
            "latex": "\\tau_\\mathrm{reio}",
        },
        "N_ur": 2.0308,
        "N_ncdm": 1,
        "Omega_gdm_max": None,
        "z_gdm_max": None,
        **{f"H_{z_idx}": None for z_idx in range(len(redshift_samples))},
    }

    def gdm_likelihood(_self):
        bg = _self.provider.get_CLASS_background()
        Omega_gdm = bg["(.)rho_gdm"] / bg["H [1/Mpc]"] ** 2
        Omega_gdm_max_idx = np.argmax(Omega_gdm)
        H0s = {
            f"H_{z_idx}": H_of_z
            for z_idx, H_of_z in zip(
                range(len(redshift_samples)),
                _self.provider.get_Hubble(redshift_samples),
            )
        }
        return (
            0,
            {
                "Omega_gdm_max": Omega_gdm[Omega_gdm_max_idx],
                "z_gdm_max": bg["z"][Omega_gdm_max_idx],
            }
            | H0s,
        )

    cobaya_info = dict(
        theory={
            "classy": {
                "extra_args": {
                    **gdm_model.fixed_settings,
                    "non_linear": "hmcode",
                    "l_max_scalars": 5000,
                },
                # "path": str(CLASS_PATH),
            }
        },
        params={**lcdm_params, **gdm_model.params},
        likelihood={
            "planck_2018_lowl.TT": None,
            "planck_2018_lowl.EE": None,
            "planck_2018_highl_plik.TTTEEE_lite": None,
            "planck_2018_lensing.clik": None,
            "gdm": {
                "external": gdm_likelihood,
                "output_params": ["Omega_gdm_max", "z_gdm_max"]
                + [f"H_{z_idx}" for z_idx in range(len(redshift_samples))],
                "requires": {
                    "CLASS_background": None,
                    "Hubble": {"z": redshift_samples},
                },
            },
        },
        sampler=dict(
            mcmc={
                "drag": True,
                "oversample_power": 0.4,
                "proposal_scale": 1.9,
                "covmat": "auto",
                "Rminus1_stop": 0.01,
                "Rminus1_cl_stop": 0.025,
            }
        ),
        output=str(CHAIN_PATH),
        packages_path=str(COBAYA_PACKAGES_PATH),
        resume=True,
    )

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    success = False

    try:
        upd_info, mcmc = run(cobaya_info)
        success = True
    except LoggedError as err:
        pass

    # Did it work? (e.g. did not get stuck)
    success = all(comm.allgather(success))

    if not success and rank == 0:
        print("Sampling failed!")


if __name__ == "__main__":
    model_file = sys.argv[1]
    main(model_file)


# %%


# %%


# %%
