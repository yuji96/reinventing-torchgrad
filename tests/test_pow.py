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


def test_4():
    """y = 6 / w = w**(-1) * 6"""
    w = Tensor(3)
    y = w / 6
    y.backward()
    assert y.value == 1 / 2
    assert w.grad == 1 / 6
