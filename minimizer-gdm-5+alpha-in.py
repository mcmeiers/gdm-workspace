# %%
import sys
from pathlib import Path

import numpy as np

from cobaya.run import run

import pandas as pd

# %% Define project name and output directory


def main(args):
    alpha_e2 = int(args[0])
    alpha = alpha_e2 * 0.01
    CLASS_PATH = "/home/mcmeiers/Projects/gdm_cosmology/code/my_class"
    log10a_knots = np.array([-14.0, -4.5, -4.0, -3.5, -3.0, -2.5, 0.0])
    w_vals_fn = 'lambda w_0, w_1, w_2, w_3, w_4: "-1.0," + ",".join(map(str, [w_0, w_1, w_2, w_3, w_4])) + ",1.0"'

    gdm_params = {
        "gdm_alpha": alpha,
        "w_0": {"prior": {"max": 1, "min": -1}, "drop": True},
        "w_1": {"prior": {"max": 1, "min": -1}, "drop": True},
        "w_2": {"prior": {"max": 1, "min": -1}, "drop": True},
        "w_3": {"prior": {"max": 1, "min": -1}, "drop": True},
        "w_4": {"prior": {"max": 1, "min": -1}, "drop": True},
        "gdm_w_vals": {"value": w_vals_fn, "derived": False},
    }

    gdm_fixed = {
        "gdm_log10a_vals": ",".join(map(str, log10a_knots)),
        "gdm_interpolation_order": 1,
        "gdm_z_alpha": 3000,
    }

    CLASS_PATH = "/home/mcmeiers/Projects/gdm_cosmology/code/my_class"
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
    }

    cobaya_info = dict(
        theory={
            "classy": {
                "extra_args": {
                    **gdm_fixed,
                    "non_linear": "hmcode",
                },
                "path": CLASS_PATH,
            }
        },
        params=lcdm_params | gdm_params,
        likelihood={
            "planck_2018_lowl.TT": None,
            "planck_2018_lowl.EE": None,
            "planck_2018_highl_plik.TTTEEE_lite": None,
            "planck_2018_lensing.clik": None,
        },
        sampler={"minimize": {"method": "scipy"}},
    )

    updated_info, sampler = run(cobaya_info)

    data = {
        k: sampler.products()["minimum"][k]
        for k in sampler.model.parameterization.sampled_params().keys()
    }
    data["gdm_alpha"] = alpha
    df = pd.DataFrame(data, index=[0])
    if alpha_e2 == 0:
        df.to_csv("min_gdn_5+1_ttteee_scipy_Nncdm.csv", mode="w", index=False)
    else:
        df.to_csv(
            "min_gdn_5+1_ttteee_scipy_Nncdm.csv", mode="a", index=False, header=False
        )


# %%

if __name__ == "__main__":
    main(sys.argv[1:])
