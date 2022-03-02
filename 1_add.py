from typing import Union


class Variable:

    def __init__(self, value):
        self.value = value

        self.grad = 1
        self.backs: list[Variable] = []

    def __eq__(self, other: "Variable") -> bool:
        if isinstance(other, Variable):
            other = other.value
        return self.value == other

    def __repr__(self):
        return self.value

    def backward(self):
        if not self.backs:
            return self.grad

        for back in self.backs:
            back.backward(back.grad * self.grad)

    def __add__(self, other) -> "Variable":
        if isinstance(other, Variable):
            next = Variable(self.value + other.value)
            next.backs.append(other)
            next.backs.append(self)
        else:
            next = Variable(self.value + other)
        return next

    __radd__ = __add__

    def __neg__(self) -> "Variable":
        self.value *= -1
        self.grad *= -1
        return self


if __name__ == "__main__":
    # y = a + 5
    a = Variable(3)
    y = a + 5
    y.backward()
    assert y == 8
    assert a.grad == 1

    # y = a + b + 5
    a = Variable(3)
    b = Variable(3)
    y = a + b + 5
    y.backward()
    assert y == 11
    assert a.grad == 1
    assert b.grad == 1

    # y = -a
    a = Variable(3)
    y = -a
    y.backward()
    assert y == -3
    assert a.grad == -1
