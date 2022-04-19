from __future__ import annotations
from typing import Dict, Union, List
import numpy as np
from copy import copy
from abc import ABC


from ruamel.yaml import YAML, yaml_object
from ruamel.yaml.comments import CommentedMap

yaml = YAML(typ="safe")
yaml.default_flow_style = None


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
    def _range_from_filter(knot_x, range_filter, default_range=(-np.inf, np.inf)):
        knot_y_range = default_range
        for filter_start, range in range_filter:
            if knot_x >= filter_start:
                knot_y_range = range
        return knot_y_range

    def __init__(self, knots_log10a, fixed_knots=None, range_filter=None):
        """
        TODO

        Args:
            knots_log10a:: Sequence(float)
              A sequence of floats where the w(log10a) values will be set
              for the spline.
            fixed_knots:: Sequence(tuple(2))
              A sequence of tuples of the form (fixed_idx,fixed_w_val)

              fixed_idx::int
                The index of knots_log10a for which the w_value is fixed.
              fixed_w_val::float
                The w_value for the fixed knot.
        """
        if fixed_knots is None:
            fixed_knots = []
        if range_filter is None:
            range_filter = []
        self.knots_log10a = tuple(knots_log10a)
        self.fixed_knots = fixed_knots
        self.n_knots = len(knots_log10a)
        self.w_dim = self.n_knots - len(self.fixed_knots)
        prefixed_w_vals = [0] * self.n_knots
        for fixed_idx, fixed_val in self.fixed_knots:
            prefixed_w_vals[fixed_idx] = fixed_val
        var_knots_idx = set(range(self.n_knots)).difference(
            {fixed_idx for (fixed_idx, _) in self.fixed_knots}
        )

        self.param_names = [f"w_{idx}" for idx in range(self.w_dim)]
        for var_idx, param in zip(var_knots_idx, self.param_names):
            prefixed_w_vals[var_idx] = param
        # Create str of the form 'w_0,w_1,...' to form the correct argument signature of the w_vals fns
        comma_separated_param_names = ",".join(map(str, self.param_names))
        # Create str of either the fixed value or the variable name with comma separation, use for output of w_val fns
        comma_separated_w_vals = ",".join(map(str, prefixed_w_vals))
        self.knots_w_vals = eval(
            f"lambda {comma_separated_param_names}:[{comma_separated_w_vals}]"
        )
        self.classy_fmt_knots_w_vals = eval(
            f"lambda {comma_separated_param_names}:','.join(map(str,[{comma_separated_w_vals}]))"
        )
        self.range_filter = range_filter
        self.range_of_param = {}
        for param, log10idx in zip(self.param_names, var_knots_idx):
            self.range_of_param[param] = wModel._range_from_filter(
                self.knots_log10a[log10idx],
                self.range_filter,
                default_range={"min": -1, "max": 1},
            )

        self.w_mins = [self.range_of_param[param]["min"] for param in self.param_names]
        self.w_maxes = [self.range_of_param[param]["max"] for param in self.param_names]

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

        self.params = {
            "gdm_alpha": {"prior": self.alpha, "latex": "\\alpha_{gdm}"},
            **{
                w_param: {"prior": self.w_model.range_of_param[w_param], "drop": True}
                for w_param in self.w_model.param_names
            },
            "gdm_w_vals": {
                "value": self.w_model.classy_fmt_knots_w_vals,
                "derived": False,
            },
        }

        self.z_alpha = z_alpha
        self.fixed_settings = {
            "gdm_log10a_vals": ",".join(map(str, self.w_model.knots_log10a)),
            "gdm_z_alpha": self.z_alpha,
            "gdm_interpolation_order": 1,
        }
        self.c_eff2 = c_eff2
        self.c_vis2 = c_vis2
        var_or_fixed_params = ["c_eff2", "c_vis2"]
        for param in var_or_fixed_params:
            if isinstance(self.__getattribute__(param), (int, float)):
                self.fixed_settings[f"gdm_{param}"] = self.__getattribute__(param)
            else:
                self.params[f"gdm_{param}"] = self.__getattribute__(param)

    @classmethod
    def to_yaml(cls, representer, instance: wModel):
        # positional arguments first
        input_param_names = cls.__init__.__code__.co_varnames[
            1 : cls.__init__.__code__.co_argcount
        ]
        input_params = {k: instance.__dict__[k] for k in input_param_names}
        return representer.represent_mapping(cls.yaml_tag, input_params)

    @classmethod
    def from_yaml(cls, constructor, node):
        dict_representation = constructor.construct_mapping(node, deep=True)
        return cls(**dict_representation)


class CosmoModel(object):

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
