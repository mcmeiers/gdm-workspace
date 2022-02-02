# %% import statements

import classy

import src.mycosmo as mc

import os
import itertools as it

import numpy as np

# %% initialize output format
cwd = os.getcwd()

output_dir = "/output/emulator-large-range/"
output_path = cwd + output_dir
try:
    os.mkdir(output_path)
except FileExistsError:
    pass

output_root_name = "emulator-large-range"
output_root_path = output_path + output_root_name

# %% initialized lcdm sampled parameters

lcdm_params = ('h', 'omega_b', 'omega_cdm', 'z_reio', 'n_s', 'ln10^{10}A_s')

range_of_lcdm_param = {'h': (0.6, 0.8),
                       "ln10^{10}A_s": (2.9, 3.1),
                       'omega_cdm': (0.07, 0.17),
                       'omega_b': (0.017, 0.027),
                       'n_s': (0.92, 1.0),
                       'z_reio': (6, 9),
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

gdm_alpha_range = (0.01, 0.2)
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

rng = np.random.default_rng()

def make_sample_params():
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
                                       output_and_pre_setting=output_and_pre_settings_DEF):
    cosmo.set({**mp.get_class_input(sample_param_vals),
               **output_and_pre_setting})
    cosmo.compute()
    cls = cosmo.lensed_cl()
    lensed_cls = np.vstack(tuple(cls[k][l_min:l_max+1].astype(float) for k in ['tt', 'te', 'ee']))
    assert np.all(lensed_cls[0,:] >= 0), 'c_l_tt not strictly positive'
    assert np.all(lensed_cls[2, :] >= 0), 'c_l_ee not strictly positive'
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


n_samples = 100

sample_sampled_params = np.empty((n_samples, n_dim_sampled))
sample_cls = np.empty((n_samples, 3, l_max-l_min+1))
sample_derived_params = np.empty((n_samples, len(derived_params)))
cosmo_errs = []
cl_errs = []

cosmo = classy.Class()
sample_count = 0
attempt_count = 0
while sample_count < n_samples and attempt_count < 2 * n_samples:
    sample_param_vals = make_sample_params()
    try:
        sample_cls[sample_count, :], sample_derived_params[sample_count, :] = \
            make_sample_cls_and_derived_params(cosmo, sample_param_vals)
        sample_sampled_params[sample_count,:] =sample_param_vals
        sample_count += 1
        attempt_count += 1
    except classy.CosmoComputationError as err:
        cosmo_errs += [(sample_param_vals, err)]
        attempt_count += 1
    except AssertionError:
        cl_errs += [sample_param_vals]
        attempt_count += 1




