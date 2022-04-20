#!/usr/bin/env python
# coding: utf-8

# In[14]:


from getdist.mcsamples import loadMCSamples
import numpy as np
import classy
from scipy.interpolate import UnivariateSpline
from pathlib import Path

OUTPUT_DIR = Path("/opt/project/output/gdm6/gdm_6_fixed_ends/gdm6_fixed_post")


# In[15]:


GDM_DATA_PATH = '/opt/project/output/gdm6/gdm_6_fixed_ends/gdm_6_fixed_ends' 
PLA_DATA_PATH = '/opt/project/output/pla_data/base/plikHM_TTTEEE_lowl_lowE_lensing/base_plikHM_TTTEEE_lowl_lowE_lensing'
gdm_samples = loadMCSamples(GDM_DATA_PATH, settings={'ignore_rows':0.5})
pla_samples = loadMCSamples(PLA_DATA_PATH, settings={'ignore_rows':0.3})


# In[16]:


pla_best_fit_sample = pla_samples.getParamSampleDict(np.argmin(pla_samples.loglikes))
gdm_best_fit_sample = gdm_samples.getParamSampleDict(np.argmin(gdm_samples.loglikes))


# In[17]:


n_H_samples = 16
n_omega_gdm_samples = 11

delta_H_frac_z_samples = 1 / np.logspace(-5, 0, n_H_samples) - 1
omega_gdm_z_samples = 1 / np.logspace(-4, -3, n_omega_gdm_samples) - 1


# In[18]:


pla_best_fit_classy_input = {
    "100*theta_s": pla_best_fit_sample["theta"],
    "omega_cdm": pla_best_fit_sample["omegach2"],
    "omega_b": pla_best_fit_sample["omegabh2"],
    "z_reio": pla_best_fit_sample["zrei"],
    "ln10^{10}A_s": pla_best_fit_sample["logA"],
    "n_s": pla_best_fit_sample["ns"],
    "output": "tCl,pCl,lCl",
    "lensing": "y",
}


# In[19]:



cosmo = classy.Class()
cosmo.set(pla_best_fit_classy_input)
cosmo.compute(["background"])

pla_best_fit_H_samples = UnivariateSpline(
    np.flip(cosmo.get_background()["z"]),
    np.flip(cosmo.get_background()["H [1/Mpc]"]),
    s=0,
)(delta_H_frac_z_samples)

cosmo.struct_cleanup()
cosmo.empty()


# In[20]:



def classy_input_from_sample(params):
    return {
        "100*theta_s": params["theta_s_1e2"],
        "omega_cdm": params["omega_cdm"],
        "omega_b": params["omega_b"],
        "z_reio": params["z_reio"],
        "ln10^{10}A_s": params["logA"],
        "n_s": params["n_s"],
        "gdm_w_vals": "-1,"
        + ",".join(map(str, (params[f"w_{widx}"] for widx in range(5))))
        + ",1",
        "gdm_alpha": params["gdm_alpha"],
        "gdm_log10a_vals": "-14.0,-4.5,-4.0,-3.5,-3.0,-2.5,0.0",
        "gdm_z_alpha": 3000,
        "gdm_interpolation_order": 1,
        "gdm_c_eff2": 1,
        "gdm_c_vis2": 0,
        "non_linear": "hmcode",
        "l_max_scalars": 5000,
        "output": "tCl,pCl,lCl",
        "lensing": "y",
    }


def get_delta_H_frac_and_omega_gdm_of_sample(sample_dict):
    classy_input = classy_input_from_sample(sample_dict)
    cosmo.set(classy_input)
    cosmo.compute(["background"])
    H_samples = UnivariateSpline(
        np.flip(cosmo.get_background()["z"]),
        np.flip(cosmo.get_background()["H [1/Mpc]"]),
        s=0,
    )(delta_H_frac_z_samples)
    Omega_gdm_samples = UnivariateSpline(
        np.flip(cosmo.get_background()["z"]),
        np.flip(
            cosmo.get_background()["(.)rho_gdm"] / cosmo.get_background()["(.)rho_tot"]
        ),
        s=0,
    )(omega_gdm_z_samples)
    cosmo.struct_cleanup()
    cosmo.empty()
    return np.concatenate(
        (
            (H_samples - pla_best_fit_H_samples) / pla_best_fit_H_samples,
            Omega_gdm_samples,
        )
    )


# In[ ]:


derived_sample = np.zeros((gdm_samples.numrows, n_H_samples + n_omega_gdm_samples))

for idx in range(gdm_samples.numrows):
    derived_sample[idx, :] = get_delta_H_frac_and_omega_gdm_of_sample(
        gdm_samples.getParamSampleDict(idx)
    )


# In[ ]:


for idx in range(n_H_samples):
    gdm_samples.addDerived(
        derived_sample[:, idx],
        f"deltaH_{delta_H_frac_z_samples[idx]:.1e}",
        f"$\deltaH({delta_H_frac_z_samples[idx]:.1e})$",
    )


# In[ ]:


for idx in range(n_omega_gdm_samples):
    gdm_samples.addDerived(
        derived_sample[:, n_H_samples + idx],
        f"Omega_gdm({omega_gdm_z_samples[idx]:.1e})",
        "$\Omega_\{gdm\}$" + f"({omega_gdm_z_samples[idx]:.1e})",
    )


# In[ ]:


gdm_samples.saveAsText('/opt/project/output/gdm6/gdm_6_fixed_ends/gdm_6_fixed_post_process')


# In[ ]:




