import copy


import classy

import numpy as np

import matplotlib.pyplot as plt

from pathlib import Path
import os

from pyrsistent import m
from functools import reduce

# %% ploting utilities


PACKAGE_DIR = Path(os.path.join(os.path.dirname(__file__)))
DATA_DIR = Path.resolve(PACKAGE_DIR / "../data/")


def plot_Cls_against_planck18(
    cls_of_mdl, ell_min=2, ell_max=2508, modes=("tt", "te", "ee"), fig=None
):
    if fig is None:
        fig, axs = plt.subplots(len(modes), sharex=True, figsize=(14, 8))
        fig.subplots_adjust(hspace=0)
    else:
        axs = fig.gca()

    if len(modes) == 1:
        axs = [axs]

    # The data file columns give Dℓ = ℓ(ℓ+1)Cℓ / 2π in units of μK2, and the lower and upper 68% uncertainties.

    assert len(modes) == len(axs)
    assert ell_min < ell_max

    for mode, ax in zip(modes, axs):
        data = np.loadtxt(
            DATA_DIR / "planck" / f"COM_PowerSpect_CMB-{mode.upper()}-full_R3.01.txt"
        )
        ell_mask = (data[:, 0] >= ell_min) & (data[:, 0] <= ell_max)
        ax.errorbar(data[ell_mask, 0], data[ell_mask, 1], yerr=data[ell_mask, 2:].T)

    labels = []
    for mdl, cls in cls_of_mdl.items():
        ell_mask = (cls["ell"] >= ell_min) & (cls["ell"] <= ell_max)
        ells = cls["ell"][ell_mask]
        labels.append(mdl)
        for mode, ax in zip(modes, axs):
            ax.plot(ells, cls[mode][ell_mask], label=mdl)
            ax.set_ylabel(
                r"$\ell(\ell+1)/(2\pi)\,C^{" + str(mode) + "}_\ell\;(\mu \mathrm{K}^2)$"
            )

    plt.xlabel(r"$\ell$")
    fig.legend(labels, loc="right")

    if len(modes) == 1:
        axs = axs[0]

    return fig


def plot_residuals_Cls_from_planck18(
    cls_of_mdl,
    ell_min=2,
    ell_max=2508,
    modes=("tt", "te", "ee"),
    fig=None,
):

    if fig is None:
        fig, axs = plt.subplots(len(modes), sharex=True, figsize=(14, 8))
        fig.subplots_adjust(hspace=0)
    else:
        axs = fig.gca()

    if len(modes) == 1:
        axs = [axs]

    assert len(modes) == len(axs)
    assert ell_min < ell_max

    ref_ells = {}
    ref_cl = {}
    var_cls = {}
    for mode in modes:
        data = np.loadtxt(
            DATA_DIR / "planck" / f"COM_PowerSpect_CMB-{mode.upper()}-full_R3.01.txt"
        )
        ref_ells[mode] = data[:, 0]
        ref_cl[mode] = data[:, 1]
        var_cls[f'-{mode}'] = data[:, 2]
        var_cls[f'+{mode}'] = data[:, 3]
    
    res_factor = {}
    ell_masks = {}
    for mode in modes:
        if mode[0]==mode[1]:
            ell_mask = (ref_ells[mode] >= ell_min) & (ref_ells[mode] <= ell_max)
            ell_masks[mode] = ell_mask
            ref_ells[mode] = ref_ells[mode][ell_mask]
            ref_cl[mode] = ref_cl[mode][ell_mask]
            var_cls[f'-{mode}'] = var_cls[f'-{mode}'][ell_mask]
            var_cls[f'+{mode}'] = var_cls[f'+{mode}'][ell_mask] 
            res_factor[mode]= 1/ref_cl[mode]
        else:
            common_ell = reduce(np.intersect1d,(ref_ells[mode],ref_ells[mode[0]+ mode[0]],ref_ells[mode[1]+ mode[1]]))
            common_ell = common_ell[(common_ell >= ell_min) & (common_ell <= ell_max)]
            ref_ells[mode], ell_masks[mode], _ = np.intersect1d(ref_ells[mode],common_ell,return_indices=True)
            ell_mask1 = np.in1d(ref_ells[mode[0]+ mode[0]],common_ell)
            ell_mask2 = np.in1d(ref_ells[mode[1]+ mode[1]],common_ell)
            res_factor[mode]= 1/(ref_cl[mode[0]+ mode[0]][ell_mask1] + ref_cl[mode[1]+ mode[1]][ell_mask2])**(1/2.)
            var_cls[f'-{mode}'] = var_cls[f'-{mode}'][ell_masks[mode]]
            var_cls[f'+{mode}'] = var_cls[f'+{mode}'][ell_masks[mode]] 
        
    ax in zip(modes, axs):
        ax.fill_between(data[ell_mask, 0], data[ell_mask, 2], data[ell_mask, 3])
        

        ref_ells[mode] = data[ell_mask, 0]
        ref_cl[mode] = data[ell_mask, 1]
        ax.fill_between(data[ell_mask, 0], data[ell_mask, 2], data[ell_mask, 3])


            
    
    labels = []
    for mdl, cls in cls_of_mdl.items():
        ell_intersection_info={mode:np.intersect1d(
                ref_ells[mode], cls["ell"], return_indices=True
            ) for mode in modes}
        for mode, ax in zip(modes, axs):
            common_ells,ref_ind,mdl_ind = ell_intersection_info[mode]
            scale_ells, scale_ell_ind0,scale_ell_ind1 = ell_intersection_info(np.intersect1d(ref_ells[mode[0] + mode[0]],ref_ells[mode[1] + mode[1]], return_indices=True)
            )
            ax.plot(
                common_ells,
                (cls[mode][mdl_ind] - ref_cl[mode][ref_ind])
                / (
                    (
                        (ref_cl[mode[0] + mode[0]][scale_ell_ind0] * ref_cl[mode[1] + mode[1]][scale_ell_ind1])
                        ** (1 / 2.0)
                    )[ref_ind]
                ),
                label=mdl,
            )

    for mode, ax in zip(modes, axs):
        ax.set_ylabel(
            r"$\Delta\,C^{"
            + str(mode)
            + "}_\ell/\,C^{"
            + str(mode)
            + "}_\ell\;(\mu \mathrm{K}^2)$"
        )

    plt.xlabel(r"$\ell$")
    fig.legend(labels, loc="right")

    if len(modes) == 1:
        axs = axs[0]

    return fig


def plot_Cls(cls_of_mdl, ell_min=2, ell_max=2508, modes=("tt", "te", "ee"), fig=None):
    if fig is None:
        fig, axs = plt.subplots(len(modes), sharex=True, figsize=(14, 8))
        fig.subplots_adjust(hspace=0)
    else:
        axs = fig.gca()

    if len(modes) == 1:
        axs = [axs]

    assert len(modes) == len(axs)
    assert ell_min < ell_max

    labels = []
    for mdl, cls in cls_of_mdl.items():
        ell_mask = (cls["ell"] >= ell_min) & (cls["ell"] <= ell_max)
        ells = cls["ell"][ell_mask]
        labels.append(mdl)
        for mode, ax in zip(modes, axs):
            ax.plot(ells, cls[mode][ell_mask], label=mdl)
            ax.set_ylabel(
                r"$\ell(\ell+1)/(2\pi)\,C^{" + str(mode) + "}_\ell\;(\mu \mathrm{K}^2)$"
            )

    plt.xlabel(r"$\ell$")
    fig.legend(labels, loc="right")

    if len(modes) == 1:
        axs = axs[0]

    return fig


def plot_residuals_Cls(
    cls_of_mdl,
    reference_mdl,
    ell_min=2,
    ell_max=2508,
    modes=("tt", "te", "ee"),
    fig=None,
):

    if fig is None:
        fig, axs = plt.subplots(len(modes), sharex=True, figsize=(14, 8))
        fig.subplots_adjust(hspace=0)
    else:
        axs = fig.gca()

    if len(modes) == 1:
        axs = [axs]

    assert len(modes) == len(axs)
    assert ell_min < ell_max

    ref_cl = cls_of_mdl[reference_mdl]
    ell_mask = (ref_cl["ell"] >= ell_min) & (ref_cl["ell"] <= ell_max)
    ref_ells = ref_cl["ell"][ell_mask]

    labels = []
    for mdl, cls in cls_of_mdl.items():
        if mdl != reference_mdl:
            assert np.all(ref_ells == cls["ell"][ell_mask])
            for mode, ax in zip(modes, axs):
                ax.plot(
                    ref_ells,
                    (cls[mode][ell_mask] - ref_cl[mode][ell_mask])
                    / (
                        ref_cl[mode[0] + mode[0]][ell_mask]
                        * ref_cl[mode[1] + mode[1]][ell_mask]
                    )
                    ** (1 / 2.0),
                    label=mdl,
                )

    for mode, ax in zip(modes, axs):
        ax.set_ylabel(
            r"$\Delta\,C^{"
            + str(mode)
            + "}_\ell/\,C^{"
            + str(mode)
            + "}_\ell\;(\mu \mathrm{K}^2)$"
        )

    plt.xlabel(r"$\ell$")
    fig.legend(labels, loc="right")

    if len(modes) == 1:
        axs = axs[0]

    return fig


# %% ModelParameters Object Definition
class ModelParameters(object):

    # instance attributes
    def __init__(
        self,
        dims_of_input_params,
        vals_of_fixed_params=None,
        fn_of_dependent_params=None,
        class_input_params=None,
    ):

        if fn_of_dependent_params is None:
            fn_of_dependent_params = {}
        if vals_of_fixed_params is None:
            vals_of_fixed_params = {}
        self.dims_of_input_params = copy.deepcopy(dims_of_input_params)
        self.input_param_names = list(self.dims_of_input_params.keys())
        self.input_splits = np.cumsum(
            [dims_of_input_params[in_param] for in_param in self.input_param_names[:-1]]
        )
        self.total_input_dim = (
            self.input_splits[-1]
            + self.dims_of_input_params[self.input_param_names[-1]]
        )

        self.vals_of_fixed_params = copy.deepcopy(vals_of_fixed_params)

        self.fn_of_dependent_params = copy.deepcopy(fn_of_dependent_params)
        self.dependent_param_of_level = None

        self.model_param_names = (
            self.input_param_names
            + list(self.vals_of_fixed_params.keys())
            + list(self.fn_of_dependent_params.keys())
        )
        # defaults to all params if class_input_params is not provided.
        self.class_input_params = class_input_params
        if self.class_input_params is None:
            self.class_input_params = self.model_param_names

    # instance method
    def get_vals_of_model_params(self, input_vals):

        vals_of_input_parameters = dict(
            zip(self.input_param_names, np.split(input_vals, self.input_splits))
        )
        vals_of_model_params = {**self.vals_of_fixed_params, **vals_of_input_parameters}

        if self.dependent_param_of_level is None:
            return self._initialize_get_vals_of_model_params(vals_of_model_params)

        for level in range(1, self._max_fn_level_ + 1):
            level_params = self.dependent_param_of_level[level]
            vals_of_model_params = {
                **vals_of_model_params,
                **dict(
                    zip(
                        level_params,
                        [
                            self.fn_of_dependent_params[param](**vals_of_model_params)
                            for param in level_params
                        ],
                    )
                ),
            }
        return vals_of_model_params

    def get_vals_of_variable_parameters(self, input_vals):

        vals_of_model_params = self.get_vals_of_model_params(input_vals)

        vals_of_variable_params = {
            k: v
            for k, v in vals_of_model_params.items()
            if k not in self.vals_of_fixed_params
        }
        return vals_of_variable_params

    def _initialize_get_vals_of_model_params(self, vals_of_model_params):
        self.dependent_param_of_level = {}
        n_dependent_vars = len(self.fn_of_dependent_params.keys())
        n_level_assigned_dependent_params = 0
        level = 1
        current_level_vars = list(self.fn_of_dependent_params.keys())
        next_level_vars = []
        current_level_values_of_vars = {}
        while n_level_assigned_dependent_params < n_dependent_vars:
            self.dependent_param_of_level[level] = []
            for var in current_level_vars:
                try:
                    current_level_values_of_vars[var] = self.fn_of_dependent_params[
                        var
                    ](**vals_of_model_params)
                    self.dependent_param_of_level[level] += [var]
                    n_level_assigned_dependent_params += 1
                except TypeError:
                    next_level_vars += [var]
            assert bool(
                current_level_values_of_vars
            ), "Variable collection unsolvable check for closure and loops"
            vals_of_model_params = {
                **vals_of_model_params,
                **current_level_values_of_vars,
            }
            current_level_vars = next_level_vars
            current_level_values_of_vars = {}
            next_level_vars = []
            level += 1

        self._max_fn_level_ = level - 1

        return vals_of_model_params

    def get_class_input(self, input_vals):
        numeric_vals_of_model_params = self.get_vals_of_model_params(input_vals)
        numeric_vals_of_class_params = {
            class_param: numeric_vals_of_model_params[class_param]
            for class_param in self.class_input_params
        }
        return self._to_class_form(numeric_vals_of_class_params)

    @staticmethod
    def _to_class_form(numeric_vals_of_params):
        class_form_vals_of_params = {}
        for param, vals in numeric_vals_of_params.items():
            if isinstance(vals, str):
                class_form_vals_of_params[param] = vals
            else:
                try:
                    class_form_vals_of_params[param] = ",".join(
                        str(val) for val in vals
                    )
                except TypeError:
                    class_form_vals_of_params[param] = vals
        return class_form_vals_of_params

    def make_class_ini(
        self,
        input_vals,
        file_name,
        output="tCl,pCl,lCl",
        lensing="yes",
        other_settings=None,
    ):

        if other_settings is None:
            other_settings = {}
        vals_of_params = self.get_class_input(input_vals)
        with open(file_name, "w") as f:
            for key, value in vals_of_params.items():
                f.write("%s=%s\n" % (key, str(value)))
            f.write("%s=%s\n" % ("output", output))
            f.write("%s=%s\n" % ("lensing", lensing))
            for key, value in other_settings.items():
                f.write("%s=%s\n" % (key, str(value)))


# %% CosmoModel Object Definition


class CosmoModel(object):
    def __init__(self, model_params: ModelParameters):

        self.model_params = model_params
        self.cosmo = classy.Class()

    def make_sample(self, input_vals, derived_params=None, post_process=None):

        classy_output_settings = {
            "output": "tCl,pCl,lCl",
            "lensing": "yes",
            "l_max_scalars": 2700,
        }
        self.cosmo.set(
            {**self.model_params.get_class_input(input_vals), **classy_output_settings}
        )
        self.cosmo.compute()
        if derived_params is not None:
            self.cosmo.get_current_derived_parameters(derived_params)


# %% CosmoModel Object Definition
