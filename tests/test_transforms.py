from scorecard.transform import ContinuousTransform, CategoricalTransform
import pytest
import numpy as np
import copy


@pytest.fixture
def x():
    return np.random.randn(10000)


@pytest.fixture
def z():
    return np.random.choice(list("abcde"), 10000, replace=True)


@pytest.fixture
def breaks():
    return [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0]


def test_continuous_constructor(x, breaks):
    v = ContinuousTransform(breaks, [], np.nan)
    assert True, "Constructor with all arguments"

    v = ContinuousTransform(breaks, [], np.nan)
    assert True, "Constructor without missing argument"

    with pytest.raises(TypeError):
        ContinuousTransform(breaks)


def test_continuous_collapse(x, breaks):
    v = ContinuousTransform(breaks, [], np.nan)

    v.collapse([0, 3])
    assert (
        v.breaks[1] == breaks[4]
    ), "collapsing first bin with 4th bin causes new first break to equal old fourth"

    v.collapse([0, 100])
    assert v.breaks == [-np.inf, np.inf], "collapsing ALL leads to (-Inf, Inf)"

    v.collapse([-10, 10])
    assert v.breaks == [-np.inf, np.inf], "collapsing ALL leads to (-Inf, Inf)"

    v.reset()
    assert v.breaks == breaks, "resetting leads to original breaks"

    v.collapse([3, 7])
    print(v.breaks, breaks[3])
    assert (
        v.breaks[3] == breaks[3]
    ), "verify collapsing right-most bins causes break value to remain the same"

    assert v.breaks[-1] == np.inf, "last break is always inf"

    assert v.breaks[0] == -np.inf, "first break is always -inf"
