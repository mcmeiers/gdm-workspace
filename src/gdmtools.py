#from ruamel.yaml import YAML
import numpy as np
from copy import copy

class wModel(object):
    """
    #TODO: model doc string
    """
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
        self.knots_log10a = list(knots_log10a)
        self._fixed_knots = fixed_knots
        self.n_knots = len(knots_log10a)
        self.w_dim = self.n_knots - len(self._fixed_knots)
        prefixed_w_vals = [0]*self.n_knots
        for fixed_idx, fixed_val in self._fixed_knots:
            prefixed_w_vals[fixed_idx] = fixed_val
        var_knots_idx = set(range(self.n_knots)).difference({fixed_idx for (fixed_idx, _) in self._fixed_knots})

        self.param_names = ['w_'+str(idx) for idx in range(self.w_dim)]
        for var_idx, param in zip(var_knots_idx,self.param_names):
            prefixed_w_vals[var_idx] = param
        cs_param_names = ','.join(map(str, self.param_names))
        cs_w_vals = ','.join(map(str, prefixed_w_vals))
        self.knots_w_vals = eval(f'lambda {cs_param_names}:[{cs_w_vals}]')
        self.cobaya_fmt_knots_w_vals = eval(f'lambda {cs_param_names}:",".join(map(str,[{cs_w_vals}]))')
        self._range_filter = range_filter
        self.range_of_param = {}
        for param,log10idx in zip(self.param_names, var_knots_idx):
            self.range_of_param[param] = _range_from_filter(self.knots_log10a[log10idx],
                                                            self._range_filter,
                                                            default_range={'min':-1,'max':1})






    # def to_yaml(self,file_path):
    #     """
    #     TODO
    #     :param file_path:
    #     :return: None
    #     """
    #     yaml=YAML(typ='safe')
    #     yaml.default_flow_style = None
    #     model_params = {'knots_log10a': self.knots_log10a}
    #     if self._fixed_knots != []:
    #         model_params['_fixed_knots'] = self._fixed_knots
    #
    #     yaml.dump(model_params, file_path)

    def w_vals(self,var_w_vals):
        """
        TODO
        :param var_w_vals:
        :return:
        """
        w_vals = copy(self._prefixed_w_vals)
        for idx, val in zip(self._var_knots_idx,var_w_vals):
            w_vals[idx] = val
        return w_vals

def _range_from_filter(knot_x, filter_range, default_range=(-np.inf,np.inf)):
    knot_y_range=default_range
    for filter_start, range in filter_range:
        if knot_x >= filter_start:
            knot_y_range = range
    return knot_y_range





# def read_w_model_from_yaml(file_path):
#     """
#
#     :param file_path:
#     :return:
#     """
#     yaml = YAML(typ='safe')
#     yaml.default_flow_style = True
#     w_model_params = yaml.load(file_path)
#     return wModel(**w_model_params)


class gdmModel:
    """TODO"""

    def __init__(self,w_mdl,alpha_range,c_eff2=0,c_vis2=0,z_alpha=0):

        self.w_model = w_mdl
        self.alpha_range = alpha_range
        self.c_eff2 = c_eff2
        self.c_vis2 = c_vis2
        self.z_alpha = z_alpha
        self.fixed_settings = {'gdm_log10a_vals': ','.join(map(str,self.w_model.knots_log10a)),
                               'gdm_c_eff2': self.c_eff2,
                               'gdm_c_vis2': self.c_vis2,
                               'gdm_z_alpha': self.z_alpha,
                               'gdm_interpolation_order': 1,
                              }

        self.cobaya_params = {'gdm_alpha': {'prior': self.alpha_range,
                              'latex': '\\alpha_{gdm}'},
                              **{w_param: {'prior': self.w_model.range_of_param[w_param],
                                           'drop': True}
                                 for w_param in self.w_model.param_names},
                              'gdm_w_vals': {'value': self.w_model.cobaya_fmt_knots_w_vals,
                                             'derived': False}
                              }
