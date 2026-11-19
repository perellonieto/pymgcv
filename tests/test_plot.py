"""Smoke tests for plotting functions, simply check that they run without errors.

To see the plots created during testing, replace plt.close("all") with plt.show()
"""

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.collections import QuadMesh

import pymgcv.plot as gplt
from pymgcv.gam import BAM, GAM, AbstractGAM
from pymgcv.terms import L

from . import gam_test_cases as tc


def get_cases_1d_continuous(gam_type: type[AbstractGAM]):
    cases = [
        (tc.linear_gam, {}),
        (tc.smooth_1d_gam, {"residuals": True}),
        (tc.smooth_1d_by_numeric_gam, {}),
        (tc.smooth_1d_random_wiggly_curve_gam, {"level": "A"}),
        (tc.smooth_1d_by_categorical_gam, {"level": "A"}),
        (tc.linear_functional_smooth_1d_gam, {}),
    ]
    return {
        f"{gam_type.__name__} - {f.__name__}": (f(gam_type), kwargs)
        for f, kwargs in cases
    }


cases_1d_continuous = get_cases_1d_continuous(GAM) | get_cases_1d_continuous(BAM)


@pytest.mark.parametrize(
    ("test_case", "kwargs"),
    cases_1d_continuous.values(),
    ids=cases_1d_continuous.keys(),
)
def test_continuous_1d(test_case: tc.GAMTestCase, kwargs: dict):
    gam = test_case.gam_model.fit(test_case.data, **test_case.fit_kwargs)
    term = list(gam.predictors.values())[0][0]
    gplt.continuous_1d(**kwargs, gam=gam, term=term, data=test_case.data)
    plt.close("all")


@pytest.mark.parametrize(
    ("test_case", "kwargs"),
    cases_1d_continuous.values(),
    ids=cases_1d_continuous.keys(),
)
def test_continuous_1d_fit_density(test_case: tc.GAMTestCase, kwargs: dict):
    gam = test_case.gam_model.fit(test_case.data, **test_case.fit_kwargs)
    term = list(gam.predictors.values())[0][0]
    gplt.continuous_1d(
        **kwargs,
        gam=gam,
        term=term,
        data=test_case.data,
        fit_density=True,
        fit_density_kwargs={"n": 20, "trans": np.sqrt, "cmap": "Greys"},
    )
    plt.close("all")


def test_fit_density_grid():
    fit = np.array([0.0, 1.0, -0.5, 2.0])
    se = np.array([0.5, 0.2, 1.0, 0.3])
    y_grid, density = gplt._fit_density_grid(fit, se, n=7, level=0.95)

    # Grid spans fit ± z * se over all x, with z = |qnorm(0.95)|
    z = 1.6448536269514722
    expected_y = np.linspace(np.min(fit - z * se), np.max(fit + z * se), 7)
    np.testing.assert_allclose(y_grid, expected_y)

    # Normal density with mean fit and sd se, rows indexed by y and columns by x
    expected_density = np.exp(
        -((y_grid[:, None] - fit) ** 2) / (2 * se**2),
    ) / (se * np.sqrt(2 * np.pi))
    assert density.shape == (7, 4)
    np.testing.assert_allclose(density, expected_density)


def test_fit_density_grid_integrates_to_one():
    # With a wide range (±6 standard errors) each column is a normal density
    fit = np.array([1.0, 1.0])
    se = np.array([0.5, 0.5])
    y_grid, density = gplt._fit_density_grid(fit, se, n=2001, level=1 - 1e-9)
    np.testing.assert_allclose(np.trapezoid(density, y_grid, axis=0), 1, rtol=1e-6)


def test_continuous_1d_fit_density_mesh():
    test_case = tc.smooth_1d_gam(GAM)
    gam = test_case.gam_model.fit(test_case.data, **test_case.fit_kwargs)
    term = gam.predictors["y"][0]

    def plot(**kwargs):
        _, ax = plt.subplots()
        gplt.continuous_1d(gam=gam, term=term, eval_density=30, ax=ax, **kwargs)
        meshes = [c for c in ax.collections if isinstance(c, QuadMesh)]
        plt.close("all")
        return meshes

    assert plot() == []  # No density unless requested

    (mesh,) = plot(fit_density=True, fit_density_kwargs={"n": 20})
    density = np.ma.getdata(mesh.get_array())
    assert density.shape == (20, 30)
    assert np.all(density >= 0)

    (mesh_trans,) = plot(
        fit_density=True,
        fit_density_kwargs={"n": 20, "trans": np.sqrt, "cmap": "Greys"},
    )
    np.testing.assert_allclose(np.ma.getdata(mesh_trans.get_array()), np.sqrt(density))
    assert mesh_trans.get_cmap().name == "Greys"


def get_cases_2d_continuous(gam_type: type[AbstractGAM]):
    test_cases_1d_continuous = [
        (tc.smooth_2d_gam, {}),
        (tc.tensor_2d_gam, {}),
        (tc.tensor_2d_by_numeric_gam, {}),
        (tc.tensor_2d_by_categorical_gam, {"level": "A"}),
        (tc.tensor_2d_random_wiggly_curve_gam, {"level": "A"}),
        (tc.linear_functional_tensor_2d_gam, {}),
    ]
    return {
        f"{gam_type.__name__} - {f.__name__}": (f(gam_type), kwargs)
        for f, kwargs in test_cases_1d_continuous
    }


cases_2d_continuous = get_cases_2d_continuous(GAM) | get_cases_2d_continuous(BAM)


@pytest.mark.parametrize(
    ("test_case", "kwargs"),
    cases_2d_continuous.values(),
    ids=cases_2d_continuous.keys(),
)
def test_continuous_2d(test_case: tc.GAMTestCase, kwargs: dict):
    gam = test_case.gam_model.fit(test_case.data, **test_case.fit_kwargs)
    term = list(gam.predictors.values())[0][0]
    gplt.continuous_2d(**kwargs, gam=gam, term=term, data=test_case.data)
    plt.close("all")


@pytest.mark.parametrize(
    "gam_type",
    [GAM, BAM],
)
def test_categorical(gam_type: type[AbstractGAM]):
    test_case = tc.categorical_linear_gam(gam_type)
    gam = test_case.gam_model.fit(test_case.data, **test_case.fit_kwargs)
    term = list(gam.predictors.values())[0][0]
    assert isinstance(term, L)
    gplt.categorical(target="y", gam=gam, term=term, data=test_case.data)
    plt.close("all")


all_test_cases = (
    cases_1d_continuous
    | cases_2d_continuous
    | {"categorical_linear": (tc.categorical_linear_gam(GAM), {"target": "y"})}
)

all_gam_test_cases = tc.get_test_cases()


@pytest.mark.parametrize(
    "test_case",
    all_gam_test_cases.values(),
    ids=all_gam_test_cases.keys(),
)
def test_plot(test_case: tc.GAMTestCase):
    gam = test_case.gam_model.fit(test_case.data, **test_case.fit_kwargs)
    try:
        gplt.plot(gam=gam, ncols=1)  # scatter=True fails for mvn + gaulss
    except (ValueError, NotImplementedError) as e:
        if "plot any" in str(e):
            pass
        else:
            raise
    plt.close("all")


@pytest.mark.parametrize(
    "test_case",
    all_gam_test_cases.values(),
    ids=all_gam_test_cases.keys(),
)
def test_qq(test_case: tc.GAMTestCase):
    gam = test_case.gam_model.fit(test_case.data, **test_case.fit_kwargs)
    try:
        gplt.qq(gam=gam)
    except NotImplementedError as e:
        accept_strs = [
            "Multivariate response",  # e.g. mvn and gaulss
            "Sample function not available",  # Quasi families
        ]
        if any(s in str(e) for s in accept_strs):
            pass
        else:
            raise

    except TypeError as e:
        if "Family must support CDF method" in str(e):
            pass
        else:
            raise
    plt.close("all")


@pytest.mark.parametrize(
    "test_case",
    all_gam_test_cases.values(),
    ids=all_gam_test_cases.keys(),
)
def test_residuals_vs_linear_predictor(test_case: tc.GAMTestCase):
    gam = test_case.gam_model.fit(test_case.data, **test_case.fit_kwargs)
    target = list(gam.predictors.keys())[0]
    try:
        gplt.residuals_vs_linear_predictor(gam, target=target)
    except NotImplementedError as e:
        if "Multivariate response" in str(e):  # e.g. mvn and gaulss
            print(gam.family.__class__.__name__)
            pass
        else:
            raise

    plt.close("all")
