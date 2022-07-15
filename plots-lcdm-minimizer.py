# %%
import sys
from pathlib import Path

import numpy as np

from cobaya.model import get_model

import pandas as pd

import matplotlib.pyplot as plt

# %% Define project name and output directory

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
    "tau_reio":{
        "prior": {"min": 0.04, "max": 0.07},
        "ref": {"dist": "norm", "loc": 0.0543, "scale": 0.0073},
        "proposal": 0.0073,
        "latex": "\\tau_\\mathrm{reio}",
    },
    "N_ur": 2.0308,
    "N_ncdm": 1
}

cobaya_info = dict(
    theory={
        "classy": {
            "extra_args": {
                "non_linear": "hmcode",
            },
            "path": CLASS_PATH,
        }
    },
    params=lcdm_params,
    likelihood={
         "planck_2018_lowl.TT": None,
         "planck_2018_lowl.EE": None,
         "planck_2018_highl_plik.TTTEEE_lite": None,
         "planck_2018_lensing.clik": None,
     },
)

model = get_model(cobaya_info)
# %%
df_lite = pd.read_csv("min_lcdm_ttteee_scipy_Nncdm.csv")

ps_lite = {k:df_lite.iloc[0][k] for k in model.parameterization.sampled_params()}

ps_full = {
    "logA":3.0448,
    "n_s":0.96605,
    "theta_s_1e2":1.041085,
    1.041783
    "omega_b":0.022383,
    "omega_cdm":0.12011,
    "tau_reio":0.0543,
    'A_planck':1
}
# %%
param_sets = [("plk18",ps_full),(r"lcdm_bf",ps_lite)]
modes = ['tt','te','ee']

Cls = {}
f, axs = plt.subplots(3, figsize=(14, 8))
for label,ps in param_sets:
    model.logposterior(ps)  # to force computation of theory
    Cls[label] = model.provider.get_Cl(ell_factor=True)
    for mode, ax in zip(modes,axs):
        ax.plot(Cls[label]["ell"][2:], Cls[label][mode][2:], label=label)
        ax.set_ylabel(r"$\ell(\ell+1)/(2\pi)\,C^"+str(mode)+"_\ell\;(\mu \mathrm{K}^2)$")
    
for ax in axs:
    ax.set_xlabel(r"$\ell$")
    ax.legend()
plt.savefig("lcdm_to_plk18.png")

# %%
f, axs = plt.subplots(3,sharex=True, figsize=(14, 8))
f.subplots_adjust(hspace=0)

axs[0].plot(Cls["lcdm_bf"]["ell"][2:],(Cls["lcdm_bf"]['tt'][2:]-Cls["plk18"]['tt'][2:])/(Cls["plk18"]['tt'][2:]), label='tt')
axs[1].plot(Cls["lcdm_bf"]["ell"][2:],(Cls["lcdm_bf"]['te'][2:]-Cls["plk18"]['te'][2:])/(Cls["plk18"]['tt'][2:]*Cls["plk18"]['ee'][2:])**(1/2.), label='te')
axs[2].plot(Cls["lcdm_bf"]["ell"][2:],(Cls["lcdm_bf"]['ee'][2:]-Cls["plk18"]['ee'][2:])/(Cls["plk18"]['ee'][2:]), label='ee')
  
for ax in axs:
    ax.set_xlabel(r"$\ell$")
    ax.legend()


plt.savefig("lcdm_to_plk18_residuals.png")



# %%
f, axs = plt.subplots(3,sharex=True, figsize=(14, 8))
f.subplots_adjust(hspace=0)

axs[0].plot(Cls["lcdm_bf"]["ell"][2:20],(Cls["lcdm_bf"]['tt'][2:20]-Cls["plk18"]['tt'][2:20])/(Cls["plk18"]['tt'][2:20]), label='tt')
axs[1].plot(Cls["lcdm_bf"]["ell"][2:20],(Cls["lcdm_bf"]['te'][2:20]-Cls["plk18"]['te'][2:20])/(Cls["plk18"]['tt'][2:20]*Cls["plk18"]['ee'][2:20])**(1/2.), label='te')
axs[2].plot(Cls["lcdm_bf"]["ell"][2:20],(Cls["lcdm_bf"]['ee'][2:20]-Cls["plk18"]['ee'][2:20])/(Cls["plk18"]['ee'][2:20]), label='ee')
  
for ax in axs:
    ax.set_xlabel(r"$\ell$")
    ax.legend()


plt.savefig("lcdm_to_plk18_residuals_lowl.png")

# %%
likesums = {}
likelihoods ={}
for label,ps in param_sets:
    likelihoods[label] =model.logposterior(ps, as_dict=True)["loglikes"] 
    likesums[label] = sum(likelihoods[label].values())
# %%
