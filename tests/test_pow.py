import math

from autograd import Tensor, exp


def test_1():
    """y = exp(x)"""
    x = Tensor(2)
    y = exp(x)
    y.backward()
    assert y.value == math.e**2
    assert x.grad == math.e**2


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
    assert y.value == math.sqrt(3), y.value
    assert math.isclose(w.grad, 1 / (2 * math.sqrt(3)))
