import numpy as np
from autograd import Tensor


def test_1():
    """y = a + a + b"""
    a = Tensor(5)
    b = Tensor(4)
    y = a + a + b
    y.backward()
    assert y.value == 14
    assert a.grad == 2
    assert b.grad == 1


def test_2():
    """y = a + a + b"""
    a = Tensor([1, 2, 3])
    b = Tensor(5)
    y = a + a + b
    y.backward()
    assert (y.value == np.array([7, 9, 11])).all()
    assert (a.grad == 2).all()
    assert (b.grad == 1).all()
