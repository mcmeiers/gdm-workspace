import numpy as np
import pytest
import src.gdmtools as gdmtools


class Test_wModel:

    def test_wModel_init_stores_knots_log10a(self):
        w_mdl = gdmtools.wModel([-1, 0])
        assert w_mdl.knots_log10a == [-1, 0]

    def test_wModel_init_counts_n_knots(self):
        w_mdl = gdmtools.wModel([-1, 0])
        assert w_mdl.n_knots == 2

    def test_wModel_init_counts_n_dim(self):
        w_mdl = gdmtools.wModel([-1, 0], fixed_knots=[(1, 1)])
        assert w_mdl.w_dim == 1

    def test_wModel_init_param_names(self):
        w_mdl = gdmtools.wModel([-1, 0])
        expected_names = ['w_0', 'w_1']
        assert w_mdl.param_names == expected_names

    def test_wModel_init_range_of_param_default(self):
        w_mdl = gdmtools.wModel([-1, 0])
        expected_range_of_param = dict(zip(['w_0', 'w_1'], ({'min': -1, 'max': 1},) * 2))
        assert w_mdl.range_of_param == expected_range_of_param

    def test_wModel_init_range_of_param_filltered(self):
        range_filter = ((-1 / 2., {'min': -1 / 2, 'max': 1 / 2}),)
        w_mdl = gdmtools.wModel([-1, 0],range_filter=range_filter)
        expected_range_of_param = {'w_0':{'min': -1, 'max': 1},
                                   'w_1':{'min': -1/2, 'max': 1/2}}
        assert w_mdl.range_of_param == expected_range_of_param

    def test_wModel_knots_w_vals_signature(self):
        w_mdl = gdmtools.wModel([-1, 0])
        assert w_mdl.knots_w_vals.__code__.co_varnames[:w_mdl.knots_w_vals.__code__.co_argcount] == ('w_0', 'w_1')

    def test_wModel_knots_w_vals_signature_with_fixed(self):
        w_mdl = gdmtools.wModel([-1, 0], fixed_knots=[(1, 1)])
        assert w_mdl.knots_w_vals.__code__.co_varnames[:w_mdl.knots_w_vals.__code__.co_argcount] == ('w_0',)

    def test_wModel_knots_w_vals_expected_output(self):
        w_mdl = gdmtools.wModel([-1, 0])
        assert all(w_mdl.knots_w_vals(1 / 2., 1 / 3.) == np.array([1 / 2., 1 / 3.]))

    def test_wModel_knots_w_vals_expected_output_fixed(self):
        w_mdl = gdmtools.wModel([-1, 0], [(0, 1)])
        assert all(w_mdl.knots_w_vals(1 / 3.) == np.array([1, 1 / 3.]))


class Test_gdmModel:

    def test_gdm_model_fixed_settings(self):
        alpha_range ={'min':0, 'max':0.3}
        c_vis2 = 1 / 5
        c_eff2 = 1 / 7
        w_mdl = gdmtools.wModel([-1, 0])
        z_alpha = 12
        gdm_mdl = gdmtools.gdmModel(w_mdl, alpha_range, c_eff2=c_eff2, c_vis2=c_vis2, z_alpha=z_alpha)
        expected_settings = {'gdm_log10a_vals': '-1,0',
                             'gdm_c_eff2': 1/7,
                             'gdm_c_vis2': 1/5,
                             'gdm_z_alpha': 12,
                             'gdm_interpolation_order': 1,
                             }
        assert gdm_mdl.fixed_settings == expected_settings

    def test_gdm_cobaya_params(self):
        w_mdl = gdmtools.wModel([-1, 0])
        alpha_range = {'min':0, 'max':0.3}
        gdm_mdl = gdmtools.gdmModel(w_mdl, alpha_range)
        expected_settings = {'gdm_alpha': {'prior': {'min':0, 'max':0.3},
                                           'latex': '\\alpha_{gdm}'},
                             'w_0': {'prior': {'min': -1, 'max': 1}, 'drop': True},
                             'w_1': {'prior': {'min': -1, 'max': 1}, 'drop': True},
                             'gdm_w_vals': {'value': 'lambda w_0,w_1:[w_0,w_1]',
                                            'derived': False}
                             }
        assert gdm_mdl.cobaya_params == expected_settings
