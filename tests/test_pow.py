import numpy as np
from autograd import Tensor, exp


def test_1():
    """y = exp(x)"""
    x = Tensor(2)
    y = exp(x)
    y.backward()
    assert y.value == np.e**2
    assert x.grad == np.e**2


def test_2():
    """y = w ** 2"""
    w = Tensor(3)
    y = w**2
    y.backward()
    assert y.value == 9
    assert w.grad == 6


def test_3():
    """y = sqrt(w)"""
    w = Tensor(3)
    y = w**0.5
    y.backward()
    assert y.value == np.sqrt(3), y.value
    assert np.isclose(w.grad, 1 / (2 * np.sqrt(3)))


def test_4():
    """y = w ** 2"""
    w = Tensor([1, 2, 3])
    y = w**2
    y.backward()
    assert (y.value == w.value**2).all()
    assert (w.grad == 2 * w.value).all()
