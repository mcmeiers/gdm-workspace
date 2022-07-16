# %%
from __future__ import annotations
from collections.abc import Callable, Collection, Sequence
from functools import wraps
from typing import Dict, Union, List
import numpy as np
from copy import copy
from abc import ABC
import inspect

import matplotlib.pyplot as plt


from ruamel.yaml import YAML, yaml_object
from ruamel.yaml.comments import CommentedMap

yaml = YAML(typ="safe")
yaml.default_flow_style = None


def make_log10a_knots_from_epochs(
    log10a_epoch_endpoints: Sequence(float), n_knots_for_interval: Sequence(int)
):
    return [
        float(x)
        for x in np.concatenate(
            tuple(
                np.linspace(start, end, n_knots, endpoint=False)
                for start, end, n_knots in zip(
                    log10a_epoch_endpoints[:],
                    log10a_epoch_endpoints[1:],
                    n_knots_for_interval,
                )
            )
            + (np.array([log10a_epoch_endpoints[-1]]),)
        )
    ]


class CosmoModelSpaceComponent(ABC):

    params: Dict[str, Union[str, float, int, List[float]]]
    fixed_params: Dict[str, Union[str, float, int, List[float]]]


@yaml_object(yaml)
class wModel:
    yaml_tag = "!wModel"
    """
    #TODO: model doc string
    """

    @staticmethod
    def _range_from_filter(
        knot_x: float,
        range_filter: Sequence(tuple[float, float]),
        default_range: dict(str, float) = {"min": -np.inf, "max": np.inf},
    ) -> tuple(float, float):
        knot_y_range = default_range
        for filter_start, range in range_filter:
            if knot_x >= filter_start:
                knot_y_range = range
        return knot_y_range

    def __init__(
        self,
        free_knots_log10a: Sequence[float],
        fixed_knots: Sequence[tuple(float, float)] = None,
        range_filter: Sequence[tuple(float, float)] = None,
    ) -> wModel:
        """
        TODO

        Args:
            knots_log10a:: Sequence(float)
              A sequence of floats where the w(log10a) values will be set
              for the spline.
            extra_fixed_knots:: Sequence(tuple(2))
              A sequence of tuples of the form (fixed_log10a_val,fixed_w_val)

              fixed_log10a_val::float
                The log10a val for which the w_value is fixed.
              fixed_w_val::float
                The w_value for the fixed knot.
        """
        if fixed_knots is None:
            fixed_knots = []
        if range_filter is None:
            range_filter = []

        # store creation variables
        self.free_knots_log10a = free_knots_log10a
        self.fixed_knots = fixed_knots
        self.range_filter = range_filter
        #

        self.w_dim = len(self.free_knots_log10a)
        self.n_knots = self.w_dim + len(self.fixed_knots)

        _log10a_param_pairs = []
        self.sampled_params = {}

        # The comma separated names of the parameters to be used to set the signature of the comma separated w_str to form the correct argument signature of the w_vals fns
        _comma_separated_param_args = ""
        for w_idx, log10a in enumerate(self.free_knots_log10a):
            param = f"w_{w_idx}"
            self.sampled_params[param] = {
                "prior": wModel._range_from_filter(
                    log10a,
                    range_filter=self.range_filter,
                    default_range={"min": -1, "max": 1},
                ),
                "drop": True,
            }
            _log10a_param_pairs.append((log10a, param))
            _comma_separated_param_args += f"{param}, "

        self.fixed_params = {}
        _comma_separated_param_kwargs = ""
        for w_idx, (log10a, w_val) in enumerate(self.fixed_knots):
            param = f"w_fixed{w_idx}"
            self.fixed_params[param] = {"value": w_val, "drop": True}
            _log10a_param_pairs.append(
                (
                    log10a,
                    param,
                )
            )
            _comma_separated_param_kwargs += f"{param}={w_val},"

        _log10a_param_pairs.sort()
        self.knots_log10a, params_in_log10a_order = tuple(zip(*_log10a_param_pairs))

        # The comma separated parameter keys ordered by log10a
        comma_separated_params_log10a_ordered = ", ".join(params_in_log10a_order)

        # Create str of either the fixed value or the variable name with comma separation, use for output of w_val fns
        self.knots_w_vals = eval(
            f"lambda {_comma_separated_param_args}{_comma_separated_param_kwargs}: [{comma_separated_params_log10a_ordered}]"
        )
        self.classy_fmt_knots_w_vals = eval(
            f"lambda {_comma_separated_param_args}{_comma_separated_param_kwargs}: ','.join(map(str,[{comma_separated_params_log10a_ordered}]))"
        )

        self.derived_params = {
            "gdm_w_vals": {"value": self.classy_fmt_knots_w_vals, "derived": False}
        }

        self.params = {
            **self.sampled_params,
            **self.fixed_params,
            **self.derived_params,
        }

    @classmethod
    def to_yaml(cls, representer, instance: wModel):
        input_param_names = cls.__init__.__code__.co_varnames[
            1 : cls.__init__.__code__.co_argcount
        ]
        input_params = {k: instance.__dict__[k] for k in input_param_names}
        return representer.represent_mapping(cls.yaml_tag, input_params)

    @classmethod
    def from_yaml(cls, constructor, node):
        dict_representation = constructor.construct_mapping(node, deep=True)
        return cls(**dict_representation)

    def w_vals(self, var_w_vals):
        """
        TODO
        :param var_w_vals:
        :return:
        """
        w_vals = copy(self._prefixed_w_vals)
        for idx, val in zip(self._var_knots_idx, var_w_vals):
            w_vals[idx] = val
        return w_vals


@yaml_object(yaml)
class gdmModel(CosmoModelSpaceComponent):
    yaml_tag = "!gdmModel"
    """TODO"""

    def __init__(self, w_model, alpha, c_eff2=0, c_vis2=0, z_alpha=0):

        self.w_model = w_model
        self.alpha = alpha
        self.c_eff2 = c_eff2
        self.c_vis2 = c_vis2
        self.z_alpha = z_alpha

        self.params = {
            "gdm_alpha": {**self.alpha, "latex": "\\alpha_{gdm}"},
            **w_model.params,
            "gdm_c_vis2": self.c_vis2,
        }
        self.has_NAP = "N"
        if c_eff2 is not None:
            self.params["c_eff2"] = self.c_eff2
            self.has_NAP = "Y"
        self.fixed_settings = {
            "gdm_log10a_vals": ",".join(map(str, self.w_model.knots_log10a)),
            "gdm_interpolation_order": 1,
            "gdm_z_alpha": self.z_alpha,
            "nap": self.has_NAP,
        }

    @classmethod
    def to_yaml(cls, representer, instance: wModel):
        # positional arguments first
        input_param_names = list(inspect.signature(cls.__init__).parameters.keys())
        input_params = {k: instance.__dict__[k] for k in input_param_names[1:]}
        return representer.represent_mapping(cls.yaml_tag, input_params)

    @classmethod
    def from_yaml(cls, constructor, node):
        dict_representation = constructor.construct_mapping(node, deep=True)
        return cls(**dict_representation)


# %%
class ModelParameters:

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
        self._1d_params = [k for k, v in self.dims_of_input_params.items() if v == 1]
        self.input_param_names = list(self.dims_of_input_params.keys())
        self.model_param_names = copy.copy(
            self.input_param_names
        )  # add input parameters to model parameters

        # Parameter values will be a concatenate vector of their values
        # the index to split alone will be the cumulative sum of dimensions
        temp_cumulative_sum = np.cumsum(
            [dims_of_input_params[in_param] for in_param in self.input_param_names]
        )
        self._input_splits = temp_cumulative_sum[:-1]
        self.total_input_dim = temp_cumulative_sum[-1]

        self.vals_of_fixed_params = copy.deepcopy(vals_of_fixed_params)
        self.model_param_names += list(
            self.vals_of_fixed_params.keys()
        )  # add input parameters to model parameters

        self.fn_of_dependent_params = copy.deepcopy(fn_of_dependent_params)
        self._dependent_param_and_required_of_level = {}
        if (
            self.fn_of_dependent_params
        ):  # if there are dependent parameters find order they need to be called in
            _requested_params_of_dependent_params = {
                param: ModelParameters._get_requested_arguments(
                    fn, self.fn_of_dependent_params.keys()
                )
                for param, fn in fn_of_dependent_params.items()
            }
            self._dependent_param_and_required_of_level = (
                ModelParameters._find_order_to_eval_dependant(
                    self.model_param_names, _requested_params_of_dependent_params
                )
            )

        self.model_param_names += list(self.fn_of_dependent_params.keys())

        # defaults to all params if class_input_params is not provided.
        self.class_input_params = class_input_params
        if self.class_input_params is None:
            self.class_input_params = self.model_param_names

    # instance method
    def get_vals_of_model_params(self, input_vals):

        vals_of_input_parameters = dict(
            zip(self.input_param_names, np.split(input_vals, self._input_splits))
        )
        for param in self._1d_params:
            vals_of_input_parameters[param] = vals_of_input_parameters[param].item()
        vals_of_model_params = {**self.vals_of_fixed_params, **vals_of_input_parameters}

        for (
            dependent_params_and_requested_params
        ) in self._dependent_param_and_required_of_level:
            for (
                dependent_param,
                requested_params,
            ) in dependent_params_and_requested_params:
                vals_of_requested_params = {
                    param: vals_of_model_params[param] for param in requested_params
                }
                vals_of_model_params[dependent_param] = self.fn_of_dependent_params[
                    dependent_param
                ](**vals_of_requested_params)

        return vals_of_model_params

    def get_vals_of_variable_parameters(self, input_vals):

        vals_of_model_params = self.get_vals_of_model_params(input_vals)

        return {
            k: v
            for k, v in vals_of_model_params.items()
            if k not in self.vals_of_fixed_params
        }

    @staticmethod
    def _get_requested_arguments(
        fn: Callable, dependent_params: Collection
    ) -> set[str]:
        return {
            param_name
            for param_name, param_info in inspect.signature(fn).parameters.items()
            if param_info.default is inspect.Parameter.empty
            or param_name in dependent_params
        }

    @staticmethod
    def _find_order_to_eval_dependant(
        provided_params: list[str],
        requested_params_of_dependent_params: dict[str, set[str]],
    ) -> dict[int, dict[str, set[str]]]:
        """Forms an ordered dictionary which informs the order the dependant parameters must be evaluated so each gets their required arguments

        Args:
            provided_params (list[str]): The parameters that fixed in model or are inputs to be provided
            requested_params_of_dependent_params (dict[str,set[str]]): A dictionary of the required arguments for each dependent parameter

        Returns:
            list[list[tuple[str,set[str]]]]: Each entry is the list of dependent parameters only using input, fixed or dependent parameters of an earlier entry
        """
        unplaced_params_and_uncalculated_arguments = copy.deepcopy(
            requested_params_of_dependent_params
        )
        current_level: int = 1
        dependent_param_of_level: list[list[str]] = [set(provided_params)]
        while unplaced_params_and_uncalculated_arguments:
            newly_calculable_params = []
            # remove last level parameters and see if you can then calculate parameter
            for (
                param,
                uncalculated_arguments,
            ) in unplaced_params_and_uncalculated_arguments.items():
                uncalculated_arguments.difference_update(
                    set(dependent_param_of_level[current_level - 1])
                )
                if not uncalculated_arguments:
                    newly_calculable_params.append(param)
            # if no newly calculable parameters found raise issue
            assert newly_calculable_params, (
                "Dependant parameters failed to close, check their function arguments and ensure closure of relations"
                + f"remaining: {unplaced_params_and_uncalculated_arguments}, found:{dependent_param_of_level}"
            )
            # remove placed parameters
            for param in newly_calculable_params:
                del unplaced_params_and_uncalculated_arguments[param]
            dependent_param_of_level.append(newly_calculable_params)
            current_level += 1

        # remove provided_params
        dependent_param_of_level.pop(0)
        return [
            [
                (param, requested_params_of_dependent_params[param])
                for param in dependent_params
            ]
            for dependent_params in dependent_param_of_level
        ]

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


# class CosmoModel(object):
#     def __init__(self, model_params):

#         self.model_params = model_params
#         self.cosmo = classy.Class()

#     def make_sample(self, input_vals, derived_params=None, post_process=None):

#         classy_output_settings = {
#             "output": "tCl,pCl,lCl",
#             "lensing": "yes",
#             "l_max_scalars": 2700,
#         }
#         self.cosmo.set(
#             {**self.model_params.get_class_input(input_vals), **classy_output_settings}
#         )
#         self.cosmo.compute()
#         if derived_params is not None:
#             self.cosmo.get_current_derived_parameters(derived_params)


# %% Ploting Utilities


def plot_Cls(cls_of_mdl, ell_min=2, ell_max=2508, modes=("tt", "te", "ee"), axs=None):
    if axs is None:
        f, axs = plt.subplots(len(modes), sharex=True, figsize=(14, 8))
        f.subplots_adjust(hspace=0)

    if len(modes) == 1:
        axs = [axs]

    assert len(modes) == len(axs)
    assert ell_min < ell_max

    for label, cls in cls_of_mdl.items():
        idx_min = np.where(cls["ell"] == ell_min)[0]
        idx_max = np.where(cls["ell"] == ell_max)[0]
        ells = cls["ell"][idx_min : idx_max + 1]
        for mode, ax in zip(modes, axs):
            ax.plot(ells, cls[mode][idx_min : idx_max + 1], label=label)
            ax.set_ylabel(
                r"$\ell(\ell+1)/(2\pi)\,C^" + str(mode) + "_\ell\;(\mu \mathrm{K}^2)$"
            )

    if len(modes) == 1:
        axs = axs[0]

    return axs


# %%
