import copy


import classy

import numpy as np
import xarray as xr

import matplotlib.pyplot as plt

import sys
import os
import git
from pathlib import Path


from functools import reduce

REPO_DIR = Path(git.Repo(".", search_parent_directories=True).working_tree_dir)
PACKAGE_DIR = Path(os.path.join(os.path.dirname(__file__)))
DATA_DIR = Path.resolve(REPO_DIR / "../data/")

# %% ploting utilities


def _load_planck18_data():
    DATA_DIR = REPO_DIR / "data"
    plk18_data_dir = DATA_DIR / "plk18_spectra/"

    modes = ["tt", "te", "ee"]
    data_sets = []
    for mode in modes:
        data = np.loadtxt(
            plk18_data_dir / f"COM_PowerSpect_CMB-{mode.upper()}-full_R3.01.txt"
        )
        ds = xr.Dataset(
            {
                "cl": xr.DataArray(
                    data[:, [1]],
                    dims=("ell", "mode"),
                    coords={"ell": data[:, 0], "mode": mode},
                ),
                "var_cl": xr.DataArray(
                    data[:, 2:],
                    dims=("ell", "var_dir"),
                    coords={"ell": data[:, 0], "var_dir": ["-", "+"]},
                ),
            }
        )
        ds["cl_binned"]
        data_sets.append(ds)
    return xr.merge(data_sets)


planck18_data = _load_planck18_data()


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
        ax.errorbar(
            planck18_data.ell.loc[ell_min : ell_max + 1],
            planck18_data.cl.loc[mode, ell_min : ell_max + 1],
            yerr=planck18_data.var_cl.loc[mode, ell_min : ell_max + 1].T,
        )

    labels = ["planck18"]
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
    spacing=1,
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

    for mode, ax in zip(modes, axs):
        ax.fill_between(
            x=planck18_data.ell.loc[ell_min : ell_max + 1],
            y1=planck18_data.var_cl.loc[mode, ell_min : ell_max + 1, "+"],
            y2=-planck18_data.var_cl.loc[mode, ell_min : ell_max + 1, "-"],
            alpha=0.3,
        )

    ref_ells = planck18_data.ell.loc[ell_min : ell_max + 1 : spacing]

    for mode, ax in zip(modes, axs):
        for mdl, cls in cls_of_mdl.items():
            ells, ref_idx, mdl_idx = np.intersect1d(
                ref_ells, cls["ell"], return_indices=True
            )
            ax.scatter(
                ells,
                (cls[mode][mdl_idx] - planck18_data.cl.loc[mode, ells]),
                label=mdl,
                s=4,
            )
        ax.set_ylabel(
            r"$\Delta\,C^{"
            + str(mode)
            + "}_\ell/\,C^{"
            + str(mode)
            + "}_\ell\;(\mu \mathrm{K}^2)$"
        )

    plt.xlabel(r"$\ell$")
    labels = ["planck18"] + list(cls_of_mdl.keys())
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
