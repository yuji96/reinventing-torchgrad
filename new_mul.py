from typing import Optional


class tensor:

    def __init__(self, value, operator: Optional["Add"] = None):
        self.value = value
        self.operator = operator
        self.grad = 0

    def __repr__(self):
        return f"tensor({self.value}, grad_fn={self.operator})"

    def __eq__(self, other):
        return self.value == other

    def backward(self, grad=1):
        self.grad += grad
        if self.operator:
            self.operator._backward(self.grad)

    def __add__(self, other):
        add = Add(self, other)
        next = tensor(add._forward(), add)
        return next

    def __mul__(self, other):
        mul = Mul(self, other)
        next = tensor(mul._forward(), mul)
        return next


class Add:

    def __init__(self, a, b):
        self.a = a if isinstance(a, tensor) else tensor(a)
        self.b = b if isinstance(b, tensor) else tensor(b)

    def __repr__(self):
        return f"<Add at 0x{id(self)}>"

    def _forward(self):
        return self.a.value + self.b.value

    def _backward(self, grad):
        self.a.backward(grad)
        self.b.backward(grad)


class Mul:

    def __init__(self, a, b):
        self.a = a if isinstance(a, tensor) else tensor(a)
        self.b = b if isinstance(b, tensor) else tensor(b)

    def __repr__(self):
        return f"<Mul at 0x{id(self)}>"

    def _forward(self):
        return self.a.value * self.b.value

    def _backward(self, grad):
        self.a.backward(grad * self.b.value)
        self.b.backward(grad * self.a.value)


if __name__ == "__main__":
    # y = 2 * a
    a = tensor(3)
    y = a * 2
    y.backward()
    assert y == 6
    assert a.grad == 2

    # y = 2 * a + 1
    a = tensor(3)
    y = a * 2 + 1
    y.backward()
    assert y == 7
    assert a.grad == 2

    # y = a * b
    a = tensor(3)
    b = tensor(5)
    y = a * b
    y.backward()
    assert y == 15
    assert a.grad == 5
    assert b.grad == 3

    # y = a * b + a
    a = tensor(3)
    b = tensor(5)
    y = a * b + a
    y.backward()
    assert y == 18
    assert a.grad == 6
    assert b.grad == 3

    # y = a * a
    a = tensor(3)
    y = a * a
    y.backward()
    assert y == 9
    assert a.grad == 6

    # y = a * a + b
    a = tensor(3)
    b = tensor(5)
    y = a * a + b
    y.backward()
    assert y == 14
    assert a.grad == 6
    assert b.grad == 1
