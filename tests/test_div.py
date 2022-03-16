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


def test_3():
    """y = 6 / w = 6 * w**-1"""
    w = Tensor([1, 2, 3])
    y = 6 / w
    y.backward()
    assert (y.value == 6 / w.value).all()
    assert (w.grad == -6 / w.value**2).all()


def test_4():
    """y = w / 6 = w * 1/6"""
    w = Tensor([1, 2, 3])
    y = w / 6
    y.backward()
    assert (y.value == w.value / 6).all()
    assert (w.grad == 1 / 6).all()
