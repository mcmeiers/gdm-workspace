# %%
from pathlib import Path
import numpy as np

import src.gdmtools as gdmtools

from cobaya.run import run
from cobaya.log import LoggedError


from mpi4py import MPI

# %% Define project name and output directory

PROJECT_NAME = "gdm_step"
PROJECT_DIR = Path.cwd() / PROJECT_NAME

CHAIN_DIR = PROJECT_DIR / "chains/"
(CHAIN_DIR / "").mkdir(exist_ok=True, parents=True)
CLASS_PATH = "/home/mcmeiers/Projects/gdm_cosmology/code/my_class"

# %% create wModel

log10a_epoch_intervals_endpoints = (-14.0, -5.0, -3.0, 0.0)
n_interval_knots = (0, 2, 1)

log10a_knots = gdmtools.make_log10a_knots_from_epochs(
    log10a_epoch_intervals_endpoints, n_interval_knots
)[:-1]
fixed_points = ((-14, 1 / 3.0), (0, 1 / 3.0))

w_range_filter = (
    (-14, {"min": -1, "max": 1 / 3.0}),
    (-7, {"min": -1, "max": 1}),
    (-2, {"min": 1 / 3, "max": 1}),
)

w_model = gdmtools.wModel(log10a_knots, fixed_points, range_filter=w_range_filter)

gdm_model = gdmtools.gdmModel(
    w_model,
    alpha={"prior": {"min": 0, "max": 0.1}},
    c_vis2=0,
    z_alpha=3000,
    has_NAP=False,
)
with open(PROJECT_DIR / f"{PROJECT_NAME}+model.yaml", "w") as f:
    gdmtools.yaml.dump(gdm_model, f)

# %% lcdm model params

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

# %% setup gdm_likelihood, it has a trivial likelihood but Omega_gdm_max and z_gdm_max are computed here


def gdm_likelihood(_self):
    bg = _self.provider.get_CLASS_background()
    Omega_gdm = bg["(.)rho_gdm"] / bg["H [1/Mpc]"] ** 2
    Omega_gdm_max_idx = np.argmax(Omega_gdm)
    H0s = {
        f"H_{z_idx}": H_of_z
        for z_idx, H_of_z in zip(
            range(len(redshift_samples)), _self.provider.get_Hubble(redshift_samples)
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


# %%

cobaya_info = dict(
    theory={
        "classy": {
            "extra_args": {
                **gdm_model.fixed_settings,
                "non_linear": "hmcode",
                "l_max_scalars": 5000,
            },
            "path": str(CLASS_PATH),
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
            "requires": {"CLASS_background": None, "Hubble": {"z": redshift_samples}},
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
    output=str(CHAIN_DIR / PROJECT_NAME),
    resume=True,
)

# %%
comm = MPI.COMM_WORLD
rank = comm.Get_rank()

if rank == 0:
    with open(PROJECT_DIR / f"{PROJECT_NAME}+model.yaml", "w") as f:
        gdmtools.yaml.dump(gdm_model, f)

success = False

try:
    upd_info, mcmc = run(cobaya_info, resume=True)
    success = True
except LoggedError as err:
    pass

# Did it work? (e.g. did not get stuck)
success = all(comm.allgather(success))

if not success and rank == 0:
    print("Sampling failed!")

# %%
