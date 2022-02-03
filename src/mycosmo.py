import copy

import numpy as np

import classy


# %% ModelParameters Object Definition
class ModelParameters(object):

    # instance attributes
    def __init__(self, dims_of_input_params, vals_of_fixed_params=None, fn_of_dependent_params=None,
                 class_input_params=None):

        if fn_of_dependent_params is None:
            fn_of_dependent_params = {}
        if vals_of_fixed_params is None:
            vals_of_fixed_params = {}
        self.dims_of_input_params = copy.deepcopy(dims_of_input_params)
        self.input_param_names = list(self.dims_of_input_params.keys())
        self.input_splits = np.cumsum([dims_of_input_params[in_param] for in_param in self.input_param_names[:-1]])
        self.total_input_dim = self.input_splits[-1] + self.dims_of_input_params[self.input_param_names[-1]]

        self.vals_of_fixed_params = copy.deepcopy(vals_of_fixed_params)

        self.fn_of_dependent_params = copy.deepcopy(fn_of_dependent_params)
        self.dependent_param_of_level = None

        self.model_param_names = self.input_param_names + list(self.vals_of_fixed_params.keys()) + list(
            self.fn_of_dependent_params.keys())
        # defaults to all params if class_input_params is not provided.
        self.class_input_params = class_input_params
        if self.class_input_params is None:
            self.class_input_params = self.model_param_names

    # instance method
    def get_vals_of_model_params(self, input_vals):

        vals_of_input_parameters = dict(zip(self.input_param_names, np.split(input_vals, self.input_splits)))
        vals_of_model_params = self.vals_of_fixed_params | vals_of_input_parameters

        if self.dependent_param_of_level is None:
            return self._initialize_get_vals_of_model_params(vals_of_model_params)

        for level in range(1, self._max_fn_level_ + 1):
            level_params = self.dependent_param_of_level[level]
            vals_of_model_params |= dict(zip(level_params,
                                             [self.fn_of_dependent_params[param](**vals_of_model_params) for param in
                                              level_params]))
        return vals_of_model_params

    def get_vals_of_variable_parameters(self, input_vals):

        vals_of_model_params = self.get_vals_of_model_params(input_vals)

        vals_of_variable_params = {k: v for k, v in vals_of_model_params.items() if k not in self.vals_of_fixed_params}
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
                    current_level_values_of_vars[var] = self.fn_of_dependent_params[var](**vals_of_model_params)
                    self.dependent_param_of_level[level] += [var]
                    n_level_assigned_dependent_params += 1
                except TypeError:
                    next_level_vars += [var]
            assert bool(
                current_level_values_of_vars), \
                "Variable collection unsolvable check for closure and loops"
            vals_of_model_params |= current_level_values_of_vars
            current_level_vars = next_level_vars
            current_level_values_of_vars = {}
            next_level_vars = []
            level += 1

        self._max_fn_level_ = level - 1

        return vals_of_model_params

    def get_class_input(self, input_vals):
        numeric_vals_of_model_params = self.get_vals_of_model_params(input_vals)
        numeric_vals_of_class_params = {class_param: numeric_vals_of_model_params[class_param] for class_param in
                                        self.class_input_params}
        return self._to_class_form(numeric_vals_of_class_params)

    @staticmethod
    def _to_class_form(numeric_vals_of_params):
        class_form_vals_of_params = {}
        for param, vals in numeric_vals_of_params.items():
            if isinstance(vals, str):
                class_form_vals_of_params[param] = vals
            else:
                try:
                    class_form_vals_of_params[param] = ','.join(str(val) for val in vals)
                except TypeError:
                    class_form_vals_of_params[param] = vals
        return class_form_vals_of_params

    def make_class_ini(self, input_vals, file_name, output='tCl,pCl,lCl', lensing='yes', other_settings=None):

        if other_settings is None:
            other_settings = {}
        vals_of_params = self.get_class_input(input_vals)
        with open(file_name, 'w') as f:
            for key, value in vals_of_params.items():
                f.write('%s=%s\n' % (key, str(value)))
            f.write('%s=%s\n' % ('output', output))
            f.write('%s=%s\n' % ('lensing', lensing))
            for key, value in other_settings.items():
                f.write('%s=%s\n' % (key, str(value)))

# %% CosmoModel Object Definition

class CosmoModel(object):

    def __init__(self, model_params: ModelParameters):

        self.model_params = model_params
        self.cosmo = classy.Class()

    def make_sample(self, input_vals, derived_params=None, post_process=None):

        classy_output_settings = {'output': 'tCl,pCl,lCl',
                                  'lensing': 'yes',
                                  'l_max_scalars': 2700}
        self.cosmo.set(self.model_params.get_class_input(input_vals) | classy_output_settings)
        self.cosmo.compute()
        if derived_params is not None:
            self.cosmo.get_current_derived_parameters(derived_params)


# %% CosmoModel Object Definition