import math

from autograd import Tensor


def test_1():
    """y = a + a + b"""
    a = Tensor(2)
    b = Tensor(4)
    y = a**b
    y.backward()
    assert y.value == 16
    assert a.grad == 32
    assert b.grad == math.log(2) * 2**4
