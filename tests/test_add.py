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
