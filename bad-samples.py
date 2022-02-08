# %%

import classy
import numpy as np
import src.mycosmo as mc
import itertools as it
from pathlib import Path


# %%
PROJECT_NAME = "bad_samples"

CWD = Path.cwd()
PROJECT_DIR = CWD / 'output' / PROJECT_NAME / ''
(PROJECT_DIR / "").mkdir(exist_ok=True, parents=True)

PLOT_PATH = PROJECT_DIR / 'plots/'
PLOT_PATH.mkdir(exist_ok=True, parents=True)

DATA_PATH = PROJECT_DIR / 'data/'
DATA_PATH.mkdir(exist_ok=True, parents=True)

# %%

lcdm_params = ('h', 'omega_b', 'omega_cdm', 'z_reio', 'n_s', 'ln10^{10}A_s')

range_of_lcdm_param = {'h': (0.6, 0.8),
                       "ln10^{10}A_s": (2.9, 3.1),
                       'omega_cdm': (0.11, 0.15),
                       'omega_b': (0.02, 0.026),
                       'n_s': (0.92, 1.0),
                       'z_reio': (6, 11),
                       }

lcdm_dim_of_params = dict(zip(lcdm_params, it.repeat(1)))

# %% initialized gdm sampled parameters

gdm_log10a_era_interval_vals = (-14.0, -7.0, -5.0, np.log10(1/501.0), 0.0)
gdm_log10a_era_interval_dims = (1, 4, 8, 1)  # the number of knots in an interval the end point is not included

gdm_log10a_intervals = []
for interval_start, interval_end, interval_dims in zip(gdm_log10a_era_interval_vals,
                                                       gdm_log10a_era_interval_vals[1:],
                                                       gdm_log10a_era_interval_dims):
    # log10a vals for the knots in the interval, interval_dim-1 accounts for interval end being start of next interval
    gdm_log10a_intervals += [np.linspace(interval_start, interval_end, interval_dims, endpoint=False)]

# add last endpoint
gdm_log10a_intervals += [np.array([gdm_log10a_era_interval_vals[-1]])]
gdm_knots_log10a = np.concatenate(gdm_log10a_intervals)

gdm_n_knots = len(gdm_knots_log10a)

z_alpha = 3000.
a_alpha = 1 / (1 + z_alpha)
alpha_gdm = 0.10
c_s2 = 1
c_vis2 = 0

gdm_vals_of_fixed_param = {'gdm_log10a_vals': gdm_knots_log10a,
                           'gdm_c_eff2': c_s2,
                           'nap': 'y',
                           'gdm_c_vis2': c_vis2,
                           'gdm_interpolation_order': 1,
                           'gdm_z_alpha': z_alpha
                           }

gdm_params = ('gdm_alpha', 'gdm_w_vals')
gdm_dims_of_params = dict(zip(gdm_params, (1, gdm_n_knots)))

gdm_alpha_range = (0.0, 0.2)
gdm_w_range = (np.array([-1.0]*(gdm_n_knots-gdm_log10a_era_interval_dims[-1])
                        +[1.0/3.0]*gdm_log10a_era_interval_dims[-1]),
               np.array([-1.0/3.0] * gdm_log10a_era_interval_dims[0]
                        + [1.0] * (gdm_n_knots - gdm_log10a_era_interval_dims[0])),
               )


def is_in_w_range(w_vals):
    return np.all(w_vals-gdm_w_range[0] >= 0) and np.all(gdm_w_range[1]-w_vals >= 0)


gdm_classy_params = ('gdm_alpha', 'gdm_w_vals') + tuple(gdm_vals_of_fixed_param.keys())

model_dim_of_params = {**lcdm_dim_of_params,**gdm_dims_of_params}
n_dim_sampled = sum(model_dim_of_params.values())

mp = mc.ModelParameters(model_dim_of_params,
                        vals_of_fixed_params=gdm_vals_of_fixed_param,
                        class_input_params=lcdm_params + gdm_classy_params)


# %%


gdm_mu_w = np.array([0] * gdm_n_knots)

dij = np.array([[np.abs(log10a1 - log10a2) for log10a1 in gdm_knots_log10a] for log10a2 in gdm_knots_log10a])

rho = 0.1
sigma = 0.75

dij_over_rho = dij / rho
Cij = (sigma ** 2 * np.exp(-dij_over_rho * dij_over_rho))

rng_def = np.random.default_rng()

def make_sample_params(rng=rng_def):
    lcdm_sample = np.array(
        [rng.uniform(*range_of_lcdm_param[param]) for param in lcdm_params])
    gmd_alpha_sample = np.array([rng.uniform(*gdm_alpha_range)])

    w_vals_sample = rng.multivariate_normal(gdm_mu_w, Cij)
    while not is_in_w_range(w_vals_sample):
        w_vals_sample = rng.multivariate_normal(gdm_mu_w, Cij)

    return np.concatenate((lcdm_sample, gmd_alpha_sample, w_vals_sample))


# %%

derived_params = ['100*theta_s', 'tau_reio', 'z_eq', 'f_gdm_max', 'Omega0_gdm']

output_and_pre_settings_DEF = {'output': 'tCl,pCl,lCl',
                               'lensing': 'yes',
                               'l_max_scalars': 5000}
l_max = 2700
l_min = 2

def get_Omega0_gdm_from_Class(cosmo:classy.Class):
    return cosmo.get_background()['(.)rho_gdm'][-1]/cosmo.get_background()['(.)rho_tot'][-1]

def make_sample_cls_and_derived_params(cosmo:classy.Class, sample_param_vals,
                                       modelparams=mp,
                                       output_and_pre_setting=None):
    if output_and_pre_setting is None:
        output_and_pre_setting = output_and_pre_settings_DEF
    cosmo.set({**modelparams.get_class_input(sample_param_vals),
               **output_and_pre_setting})
    cosmo.compute()
    cls = cosmo.lensed_cl()
    lensed_cls = np.vstack(tuple(cls[k][l_min:l_max+1].astype(float) for k in ['tt', 'te', 'ee']))
    vals_of_derived_params = {**{k: np.array([v], dtype=float) for k, v in
                              cosmo.get_current_derived_parameters(['100*theta_s', 'tau_reio']).items()},
                              **dict(z_eq=np.array([cosmo.z_eq()], dtype=float),
                                     f_gdm_max=np.array([np.max(cosmo.get_background()['f_gdm'].astype(float))]),
                                     Omega0_gdm=np.array([get_Omega0_gdm_from_Class(cosmo).astype(float)]))}

    derived_param_vals = np.concatenate(tuple(vals_of_derived_params[param] for param in derived_params))
    cosmo.struct_cleanup()
    cosmo.empty()

    return lensed_cls, derived_param_vals



# %%

cosmo = classy.Class()

def make_bad_samples(n_total=100):

    rng = np.random.default_rng(1000)

    bad_sampled_params = np.empty((n_total, n_dim_sampled))
    bad_sample_cls = np.empty((n_total, 3, l_max - l_min + 1))
    bad_sample_derived_params = np.empty((n_total, len(derived_params)))

    cosmo = classy.Class()
    n_bad_samples = 0
    attempt_count = 0
    while n_bad_samples < n_total and attempt_count < 100 * n_total:
        sample_param_vals = make_sample_params(rng=rng)
        try:
            bad_sample_cls[n_bad_samples, :], bad_sample_derived_params[n_bad_samples, :] = \
                make_sample_cls_and_derived_params(cosmo, sample_param_vals)
            if  np.any(bad_sample_cls[n_bad_samples, 0,:] < 0) or  np.any(bad_sample_cls[n_bad_samples, 2, :] < 0):
               bad_sampled_params[n_bad_samples, :] = sample_param_vals
               n_bad_samples += 1
            attempt_count += 1
        except classy.CosmoComputationError:
            attempt_count += 1

    assert n_bad_samples==n_total, "Bad samples at rate < 0.01"

    return bad_sampled_params, bad_sample_cls, bad_sample_derived_params



# %%

bad_params, bad_cls, bad_derived_params = make_bad_samples(n_total=1)

# %%

precise_mp = mc.ModelParameters(model_dim_of_params,
                                vals_of_fixed_params={**gdm_vals_of_fixed_param, 'tol_perturbations_integration':1e-6},
                                class_input_params=lcdm_params + gdm_classy_params)


# %%
np.max(make_sample_cls_and_derived_params(cosmo,bad_params[0],modelparams=precise_mp)[0]-bad_cls[0])

# %%

make_sample_cls_and_derived_params[]

bad_cls[0,2,2188]
