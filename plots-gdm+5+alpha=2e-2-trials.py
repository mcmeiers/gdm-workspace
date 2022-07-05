# %%
from re import A
import src.mycosmo as mc

import numpy as np

from cobaya.model import get_model

import pandas as pd

import matplotlib.pyplot as plt

# %% Define project name and output directory

MODEL_NAME = "gdm-5+alpha=2e-2"

# %% Define project name and output directory

CLASS_PATH = "/home/mcmeiers/Projects/gdm_cosmology/code/my_class"
log10a_knots = np.array([-14.0, -4.5, -4.0, -3.5, -3.0, -2.5, 0.0])
w_vals_fn = lambda w_0, w_1, w_2, w_3, w_4: [-1.0, w_0, w_1, w_2, w_3, w_4, 1.0]
w_vals_fn_classy = 'lambda w_0, w_1, w_2, w_3, w_4: "-1.0," + ",".join(map(str, [w_0, w_1, w_2, w_3, w_4])) + ",1.0"'

gdm_params = {
    "gdm_alpha": {"prior": {"max": 0.1, "min": 0}},
    "w_0": {"prior": {"max": 1, "min": -1}, "drop": True},
    "w_1": {"prior": {"max": 1, "min": -1}, "drop": True},
    "w_2": {"prior": {"max": 1, "min": -1}, "drop": True},
    "w_3": {"prior": {"max": 1, "min": -1}, "drop": True},
    "w_4": {"prior": {"max": 1, "min": -1}, "drop": True},
    "gdm_w_vals": {"value": w_vals_fn_classy, "derived": False},
}

gdm_fixed = {
    "gdm_log10a_vals": ",".join(map(str, log10a_knots)),
    "gdm_interpolation_order": 1,
    "gdm_z_alpha": 3000,
}

CLASS_PATH = "/home/mcmeiers/Projects/gdm_cosmology/code/my_class/"
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
            # "path": CLASS_PATH,
        }
    },
    params=lcdm_params | gdm_params,
    likelihood={
        "planck_2018_lowl.TT": None,
        "planck_2018_lowl.EE": None,
        "planck_2018_highl_plik.TTTEEE_lite": None,
        "planck_2018_lensing.clik": None,
    },
)

model = get_model(cobaya_info)
# %%
df = pd.read_csv("min_gdm_alpha=2e-2-trials.csv")
BL_df = pd.read_csv("min_gdn_5+1_ttteee_scipy_Nncdm.csv")
param_sets = {
    row["run"]: {k: row[k] for k in model.parameterization.sampled_params()}
    for _, row in df.iterrows()
}
# %%
param_sets["BL"] = {
    k: BL_df.loc[BL_df["gdm_alpha"] == 0.0][k].item()
    for k in model.parameterization.sampled_params()
}
# %%
modes = ["tt", "te", "ee"]
redshifts = np.linspace(0, 6, 40)
model.add_requirements({"Hubble": {"z": redshifts}, "CLASS_background": None})

# %%
Cls = {}
likesums = {}
likelihoods = {}
H = {}
bgs = {}
for k, ps in param_sets.items():
    likelihoods[k] = model.logposterior(ps, as_dict=True)["loglikes"]
    likesums[k] = sum(likelihoods[k].values())
    Cls[k] = model.provider.get_Cl(ell_factor=True)
    H[k] = model.provider.get_Hubble(redshifts)
    bgs[k] = model.provider.get_CLASS_background()

# %%


mc.plot_Cls(Cls)
plt.savefig(f"{MODEL_NAME}_cls-test.png")

# %%


mc.plot_Cls(Cls, ell_max=30)
plt.savefig(f"{MODEL_NAME}_cls_lowl.png")


# %%

mc.plot_residuals_Cls(Cls, 0.0)

plt.savefig(f"{MODEL_NAME}_residuals.png")


# %%

fig, axs = plt.subplots(1, figsize=(14, 8))


labels = []
for k, ps in param_sets.items():
    if k != "BL":
        axs.plot(log10a_knots, w_vals_fn(*[ps[f"w_{idx}"] for idx in range(5)]))
        labels.append(f"trial={int(k)}")

fig.legend(labels, loc="right")
plt.xlabel(r"$\log a$")


# %%


f, ax_h = plt.subplots(1, figsize=(14, 6))
for k in param_sets.keys():
    ax_h.plot(redshifts, H[k], label=f"trial={k}")
ax_h.set_ylabel(r"$H\;(\mathrm{km}/\mathrm{s}/\mathrm{Mpc}^{-1})$")
ax_h.set_xlabel(r"$z$")
ax_h.legend()

# %%


f, ax_h = plt.subplots(1, figsize=(14, 6))
for k in param_sets.keys():
    if k != "BL":
        ax_h.plot(redshifts, (H[k] - H["BL"]) / H["BL"], label=f"trial={k}")

ax_h.set_xlabel(r"$z$")
ax_h.legend()
# %%
# %%


f, ax_h = plt.subplots(1, figsize=(14, 6))
for k in param_sets.keys():
    if k != "BL":
        zmask = (bgs[k]["z"] >= 0) & (bgs[k]["z"] <= 10**7)
        ax_h.plot(
            np.log(bgs[k]["z"][zmask]),
            (bgs[k]["(.)rho_gdm"] / bgs[k]["H [1/Mpc]"] ** 2)[zmask],
            label=f"trial={k}",
        )

ax_h.set_xlabel(r"$z$")
ax_h.legend()
# %%
