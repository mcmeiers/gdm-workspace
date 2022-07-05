import numpy as np
import pytest
import inspect
import src.gdmtools as gdmtools


class Test_wModel:
    def test_wModel_init_stores_knots_log10a(self):
        w_mdl = gdmtools.wModel((-1, 0))
        assert w_mdl.free_knots_log10a == (-1, 0)

    def test_wModel_init_counts_n_knots(self):
        w_mdl = gdmtools.wModel([-1, 0])
        assert w_mdl.n_knots == 2

    def test_wModel_init_counts_n_dim(self):
        w_mdl = gdmtools.wModel([-1], fixed_knots=[(0, 1)])
        assert w_mdl.w_dim == 1

    def test_wModel_init_free_params_keys(self):
        w_mdl = gdmtools.wModel([-1, 0])
        expected_names = ["w_0", "w_1"]
        assert list(w_mdl.sampled_params.keys()) == expected_names

    def test_wModel_init_free_params_values(self):
        w_mdl = gdmtools.wModel([-1, 0])
        expected_params = dict(
            zip(["w_0", "w_1"], ({"prior": {"min": -1, "max": 1}, "drop": True},) * 2)
        )
        assert w_mdl.sampled_params == expected_params

    def test_wModel_init_free_params_values_with_range_filter(self):
        range_filter = ((-1 / 2.0, {"min": -1 / 2, "max": 1 / 2}),)
        w_mdl = gdmtools.wModel([-1, 0], range_filter=range_filter)
        expected_params = {
            "w_0": {"prior": {"min": -1, "max": 1}, "drop": True},
            "w_1": {"prior": {"min": -1 / 2, "max": 1 / 2}, "drop": True},
        }
        assert w_mdl.sampled_params == expected_params

    def test_wModel_init_free_params_values_with_fixed(self):
        w_mdl = gdmtools.wModel([-1], fixed_knots=[(0, 1)])
        expected_params = {"w_0": {"prior": {"min": -1, "max": 1}, "drop": True}}
        assert w_mdl.sampled_params == expected_params

    def test_wModel_init_fixed_params_values(self):
        w_mdl = gdmtools.wModel([-1], fixed_knots=[(0, 1)])
        expected_params = {"w_fixed0": {"value": 1, "drop": True}}
        assert w_mdl.fixed_params == expected_params

    def test_wModel_knots_w_vals_expected_output(self):
        w_mdl = gdmtools.wModel([-1, 0])
        assert all(w_mdl.knots_w_vals(1 / 2.0, 1 / 3.0) == np.array([1 / 2.0, 1 / 3.0]))

    def test_wModel_knots_w_vals_expected_output_with_fixed(self):
        w_mdl = gdmtools.wModel([-1], fixed_knots=[(0, 1)])
        assert all(w_mdl.knots_w_vals(1 / 3.0) == np.array([1 / 3.0, 1]))

    def test_wModel_classy_fmt_knots_w_vals_signature(self):
        w_mdl = gdmtools.wModel([-1, 0])
        assert tuple(inspect.signature(w_mdl.knots_w_vals).parameters.keys()) == (
            "w_0",
            "w_1",
        )

    def test_wModel_classy_fmt_knots_w_vals_args_with_fixed(self):
        w_mdl = gdmtools.wModel([0], fixed_knots=[(-1, 1)])
        expected_args = ("w_0",)
        assert (
            tuple(
                param_name
                for param_name, param_info in inspect.signature(
                    w_mdl.knots_w_vals
                ).parameters.items()
                if param_info.default == inspect.Parameter.empty
            )
            == expected_args
        )

    def test_wModel_classy_fmt_knots_w_vals_expected_output(self):
        w_mdl = gdmtools.wModel([-1, 0])
        assert w_mdl.classy_fmt_knots_w_vals(1 / 2.0, 1 / 3.0) == f"{1/2.},{1/3.}"

    def test_wModel_classy_fmt_knots_w_vals_expected_output_with_fixed(self):
        w_mdl = gdmtools.wModel([-1], [(0, 1)])
        assert w_mdl.classy_fmt_knots_w_vals(1 / 3.0) == f"{1 / 3.0},1"


class Test_gdmModel:
    def test_gdm_model_fixed_settings(self):
        w_mdl = gdmtools.wModel([-1, 0])
        alpha = {"min": 0, "max": 0.3}
        gdm_mdl = gdmtools.gdmModel(w_mdl, alpha)
        expected_settings = {
            "gdm_log10a_vals": "-1,0",
            "gdm_interpolation_order": 1,
        }
        assert gdm_mdl.fixed_settings == expected_settings

    def test_gdm_params(self):
        w_mdl = gdmtools.wModel([-1, 0])
        alpha = {"min": 0, "max": 0.3}
        gdm_mdl = gdmtools.gdmModel(w_mdl, alpha)
        expected_settings = {
            "gdm_alpha": {"prior": {"min": 0, "max": 0.3}, "latex": "\\alpha_{gdm}"},
            "w_0": {"prior": {"min": -1, "max": 1}, "drop": True},
            "w_1": {"prior": {"min": -1, "max": 1}, "drop": True},
            "gdm_w_vals": {"value": w_mdl.classy_fmt_knots_w_vals, "derived": False},
            "gdm_c_eff2": 0,
            "gdm_c_vis2": 0,
            "gdm_z_alpha": 0,
        }
        assert gdm_mdl.params == expected_settings

    def test_gdm_params_with_variable_non_defaults(self):
        w_mdl = gdmtools.wModel([-1, 0])
        alpha = {"min": 0, "max": 0.3}
        c_vis2 = 1 / 5
        c_eff2 = {"min": 0, "max": 1}
        w_mdl = gdmtools.wModel([-1, 0])
        z_alpha = 12
        gdm_mdl = gdmtools.gdmModel(
            w_mdl, alpha, c_eff2={"min": 0, "max": 1}, c_vis2=c_vis2, z_alpha=z_alpha
        )
        expected_settings = {
            "gdm_alpha": {"prior": {"min": 0, "max": 0.3}, "latex": "\\alpha_{gdm}"},
            "w_0": {"prior": {"min": -1, "max": 1}, "drop": True},
            "w_1": {"prior": {"min": -1, "max": 1}, "drop": True},
            "gdm_w_vals": {"value": w_mdl.classy_fmt_knots_w_vals, "derived": False},
            "gdm_c_eff2": c_eff2,
            "gdm_c_vis2": c_vis2,
            "gdm_z_alpha": z_alpha,
        }
        assert gdm_mdl.params == expected_settings
