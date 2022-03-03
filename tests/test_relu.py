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
