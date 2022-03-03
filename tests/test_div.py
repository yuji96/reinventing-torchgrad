from autograd import Tensor


def test_1():
    """y = 6 / w = 6 * w**-1"""
    w = Tensor(3)
    y = 6 / w
    y.backward()
    assert y.value == 2
    assert w.grad == -2 / 3


def test_2():
    """y = w / 6 = w * 1/6"""
    w = Tensor(3)
    y = w / 6
    y.backward()
    assert y.value == 1 / 2
    assert w.grad == 1 / 6
