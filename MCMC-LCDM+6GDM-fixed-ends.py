import numpy as np

from pathlib import Path

import src.gdmtools as gdmtools

from cobaya.run import run
from cobaya.model import get_model
from cobaya.log import LoggedError

from mpi4py import MPI

# %% Define project name and output directory
PROJECT_NAME = "gdm_6_fixed_ends"

OUTPUT_DIR = Path("/opt/project/output") / PROJECT_NAME
(OUTPUT_DIR / "").mkdir(exist_ok=True, parents=True)

COBAYA_PACKAGES_PATH = Path("/software/cobaya_packages")
CLASS_PATH = COBAYA_PACKAGES_PATH / "code/class_public-designer/"

# %% create wModel

log10a_epoch_intervals_endpoints = (-14.0, -4.5, -2.5, 0.0)
n_inter_interval_knots = (0, 3, 0)

log10a_knots = np.concatenate(
    tuple(
        np.linspace(start, end, n_knots + 1, endpoint=False)
        for start, end, n_knots in zip(
            log10a_epoch_intervals_endpoints[:],
            log10a_epoch_intervals_endpoints[1:],
            n_inter_interval_knots,
        )
    )
    + (np.array([log10a_epoch_intervals_endpoints[-1]]),)
)

fixed_points = ((0, -1), (len(log10a_knots) - 1, 1))

w_range_filter = (
    (-14, {"min": -1, "max": 1 / 3.0}),
    (-7, {"min": -1, "max": 1}),
    (-2, {"min": 1 / 3, "max": 1}),
)

w_model = gdmtools.wModel(log10a_knots, fixed_points, range_filter=w_range_filter)

# %% create gdm model

alpha_range = {"min": 0, "max": 0.2}
gdm_model = gdmtools.gdmModel(w_model, alpha_range, c_eff2=1, c_vis2=0, z_alpha=3000)
gdm_cobaya_params = gdm_model.cobaya_params
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
                **gdm_model.fixed_settings,
                "non_linear": "hmcode",
                "l_max_scalars": 5000,
            },
            "path": str(CLASS_PATH),
        }
    },
    params={**lcdm_params, **gdm_cobaya_params},
    likelihood={
        "planck_2018_lowl.TT": None,
        "planck_2018_lowl.EE": None,
        "planck_2018_highl_plik.TTTEEE_lite": None,
        "planck_2018_lensing.clik": None,
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
    output=str(OUTPUT_DIR),
    packages_path=str(COBAYA_PACKAGES_PATH),
)

# %%

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

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
