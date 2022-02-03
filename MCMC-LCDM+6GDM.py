import numpy as np

from pathlib import Path

import src.mycosmo as mc

# %%
PROJECT_NAME = "gdm_5+1"

CWD = Path.cwd()
PROJECT_DIR = CWD / 'output' / PROJECT_NAME / ''
(PROJECT_DIR / "").mkdir(exist_ok=True, parents=True)

PLOT_PATH = PROJECT_DIR / 'plots/'
PLOT_PATH.mkdir(exist_ok=True, parents=True)

DATA_PATH = PROJECT_DIR / 'data/'
DATA_PATH.mkdir(exist_ok=True, parents=True)

COBAYA_PACKAGES_PATH = CWD.parent
CLASS_PATH = COBAYA_PACKAGES_PATH / 'code/my_class/'

# %% in this cell: initialize gdm model

# initialize w model
log10a_epoch_intervals_endpoints = (-14.0, -4.0, -2.0, 0.0)
epoch_intervals_n_knots = (2, 3, 1)

def piecewise_linspace(interval_endpoints: iter(), interval_n_knots: :
    """

    :param interval_endpoints:
    :param interval_knots:
    :return:
    """


n_knots = sum(epoch_intervals_n_knots)

epoch_log10a_knots = tuple(tuple(np.linspace(start, end, n_knots, endpoint=True))
                           for start, end, n_knots in zip(log10a_epoch_intervals_endpoints[:-1],
                                                          log10a_epoch_intervals_endpoints[1:],
                                                          epoch_intervals_n_knots))
# %%
log10a_knots = sum(np.linspace(start, end, n_knots, endpoint=False)
                   for start, end, n_knots in zip(log10a_epoch_intervals_endpoints[:-1],
                                                  log10a_epoch_intervals_endpoints[1:],
                                                  epoch_intervals_n_knots), (,) ) + \
               tuple(np.linspace(log10a_epoch_intervals_endpoints[-2],
                                   log10a_epoch_intervals_endpoints[-1]
                                                 epoch_intervals_n_knots[-1]-1, endpoint=False))))

# store properties of gdm_fluid
c_s2 = 1
c_vis2 = 0
z_alpha = 3000.

gdm_fixed_setting_classy = {'gdm_log10a_vals': ','.join([str(val) for val in log10a_knots]),
                            'gdm_c_eff2': c_s2,
                            'gdm_c_vis2': c_vis2,
                            'gdm_z_alpha': z_alpha
                            }

w_ranges_of_epoch = {'early': dict(min=-1.0, max=1.0 / 3.0),
                     'transition': dict(min=-1.0, max=1.0),
                     'late': dict(min=1.0 / 3.0, max=1.0)}


def w_range_for_log10a(log10a):
    epoch = 'early'
    if log10a < log10a_epoch_intervals_endpoints[1]:
        pass
    elif log10a < log10a_epoch_intervals_endpoints[2]:
        epoch = 'transition'
    else:
        epoch = 'late'
    return w_ranges_of_epoch[epoch]


w_params = tuple('w_' + str(idx) for idx in range(n_knots))

range_of_w_param = {w_param: w_range_for_log10a(log10a_knot)
                    for w_param, log10a_knot in zip(w_params, log10a_knots)}

w_vals_classy_for_cobaya = "lambda " + "=0,".join(w_params) + "=0" + \
                           ":','.join([str(val) for val in (" + ','.join(w_params) + ")])"

# %%
alpha_gdm_range = {'min': 0, 'max': 0.2}

gdm_cobaya_params = {'gdm_alpha': {'prior': alpha_gdm_range,
                                   'latex': '\\alpha_{gdm}'},
                     **{w_param: {'prior': range_of_w_param[w_param],
                                  'drop': True}
                        for w_param in w_params},
                     'gdm_w_vals': {'value': w_vals_classy_for_cobaya,
                                    'derived': False}
                     }

# %%


cobaya_input = dict(theory={'classy': {'extra_args': gdm_fixed_setting_classy,
                                       'path': str(CLASS_PATH)}},
                    params={'logA': {'prior': {'min': 1.61, 'max': 3.91},
                                     'ref': {'dist': 'norm', 'loc': 3.05,
                                             'scale': 0.001},
                                     'proposal': 0.001,
                                     'latex': '\\log(10^{10} A_\\mathrm{s})',
                                     'drop': True},
                            'A_s': {
                                'value': 'lambda logA: 1e-10*np.exp(logA)',
                                'latex': 'A_\\mathrm{s}'},
                            'n_s': {'prior': {'min': 0.8, 'max': 1.2},
                                    'ref': {'dist': 'norm', 'loc': 0.965,
                                            'scale': 0.004},
                                    'proposal': 0.002,
                                    'latex': 'n_\\mathrm{s}'},
                            'h': {'prior': {'min': 0.20, 'max': 1},
                                  'ref': {'dist': 'norm', 'loc': 0.67,
                                          'scale': 0.02},
                                  'proposal': 0.02,
                                  'latex': 'h'},
                            'omega_b': {'prior': {'min': 0.005, 'max': 0.1},
                                        'ref': {'dist': 'norm',
                                                'loc': 0.0224,
                                                'scale': 0.0001},
                                        'proposal': 0.0001,
                                        'latex': '\\Omega_\\mathrm{b} h^2'},
                            'omega_cdm': {
                                'prior': {'min': 0.001, 'max': 0.99},
                                'ref': {'dist': 'norm', 'loc': 0.12,
                                        'scale': 0.001},
                                'proposal': 0.0005,
                                'latex': '\\Omega_\\mathrm{c} h^2'},
                            'tau_reio': {'prior': {'min': 0.01, 'max': 0.8},
                                         'ref': {'dist': 'norm',
                                                 'loc': 0.055,
                                                 'scale': 0.006},
                                         'proposal': 0.003,
                                         'latex': '\\tau_\\mathrm{reio}'},
                            **gdm_cobaya_params
                            },
                    likelihood={'planck_2018_lowl.TT': None,
                                'planck_2018_lowl.EE': None,
                                'planck_2018_highl_plik.TTTEEE_lite': None},
                    sampler=dict(mcmc={"drag": True,
                                       "oversample_power": 0.4,
                                       "proposal_scale": 1.9,
                                       "covmat": "auto",
                                       "Rminus1_stop": 0.2,
                                       "Rminus1_cl_stop": 0.5}),
                    output=str(DATA_PATH / PROJECT_NAME),
                    packages_path = COBAYA_PACKAGES_PATH
                    )

# %%

from cobaya.run import run

updated_info, sampler = run(cobaya_input, debug=True, stop_at_error=True)