import numpy as np
from autograd import Tensor, log


def test_1():
    """y = log(x)"""
    w = Tensor(3)
    y = log(w)
    y.backward()
    assert y.value == np.log(3)
    assert w.grad == 1 / 3


def test_2():
    """y = log(2x)"""
    w = Tensor(3)
    y = log(2 * w)
    y.backward()
    assert y.value == np.log(6)
    assert w.grad == 1 / 3


def test_3():
    """y = log(x**2)"""
    w = Tensor(3)
    y = log(w**2)
    y.backward()
    assert y.value == 2 * np.log(3)
    assert w.grad == 2 / 3


def test_4():
    """y = log(x)"""
    w = Tensor([1, 2, 3])
    y = log(w)
    y.backward()
    assert (y.value == np.log([1, 2, 3])).all()
    assert (w.grad == 1 / w.value).all()
