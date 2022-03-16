from autograd import Tensor


def test_1():
    """y = a - 1"""
    a = Tensor(5)
    y = a - 1
    y.backward()
    assert y.value == 4
    assert a.grad == 1


def test_2():
    """y = 1 - a"""
    a = Tensor(5)
    y = 1 - a
    y.backward()
    assert y.value == -4
    assert a.grad == -1


def test_3():
    """y = a + a - b"""
    a = Tensor(5)
    b = Tensor(4)
    y = a + a - b
    y.backward()
    assert y.value == 6
    assert a.grad == 2
    assert b.grad == -1


def test_4():
    """y = 1 - a"""
    a = Tensor([1, 2, 3])
    y = 1 - a
    y.backward()
    assert (y.value == 1 - a.value).all()
    assert (a.grad == -1).all()
