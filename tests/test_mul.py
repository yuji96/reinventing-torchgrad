from autograd import Tensor


def test_1():
    """y = 2 * a"""
    a = Tensor(3)
    y = a * 2
    y.backward()
    assert y.value == 6
    assert a.grad == 2


def test_2():
    """y = 2 * a + 1"""
    a = Tensor(3)
    y = a * 2 + 1
    y.backward()
    assert y.value == 7
    assert a.grad == 2


def test_3():
    """y = a * b"""
    a = Tensor(3)
    b = Tensor(5)
    y = a * b
    y.backward()
    assert y.value == 15
    assert a.grad == 5
    assert b.grad == 3


def test_4():
    """y = a * b + a"""
    a = Tensor(3)
    b = Tensor(5)
    y = a * b + a
    y.backward()
    assert y.value == 18
    assert a.grad == 6
    assert b.grad == 3


def test_5():
    """y = a * a"""
    a = Tensor(3)
    y = a * a
    y.backward()
    assert y.value == 9
    assert a.grad == 6


def test_6():
    """y = a * a + b"""
    a = Tensor(3)
    b = Tensor(5)
    y = a * a + b
    y.backward()
    assert y.value == 14
    assert a.grad == 6
    assert b.grad == 1


def test_7():
    """y = a * a + b"""
    a = Tensor([1, 2, 3])
    b = Tensor([4, 5, 6])
    y = a * a + b
    y.backward()
    assert (y.value == a.value ** 2 + b.value).all()
    assert (a.grad == 2 * a.value).all()
    assert (b.grad == 1).all()
