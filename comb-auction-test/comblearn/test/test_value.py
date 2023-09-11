from comblearn.data import DSFValueFunction
from comblearn.env import BundleGenerator

def test_dsf_value_function():
    m = 10
    dsf = DSFValueFunction(list(range(m)), 100, [3, 2], 500)
    bgen = BundleGenerator(list(range(m)))
    x = dsf(bgen(6))
    assert x.shape == (6, 1)