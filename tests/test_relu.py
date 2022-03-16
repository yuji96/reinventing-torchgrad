import numpy as np
from autograd import Tensor
from autograd.transformers import relu


def test_1():
    """y = relu(+)"""
    x = Tensor(3)
    y = relu(x)
    y.backward()
    assert y.value == 3
    assert x.grad == 1


def test_2():
    """y = relu(-)"""
    x = Tensor(-3)
    y = relu(x)
    y.backward()
    assert y.value == 0
    assert x.grad == 0


def test_3():
    """y = relu(-+)"""
    x = Tensor([-2, 2])
    y = relu(x)
    y.backward()
    assert (y.value == np.array([0, 2])).all()
    assert (x.grad == np.array([0, 1])).all()
