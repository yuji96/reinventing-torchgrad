from numbers import Number
from typing import Union


class Variable:

    def __init__(self, value: Number):
        self.value = value

        self.grad = 1
        self.next: Variable = None

    def __eq__(self, other: "Variable") -> bool:
        if isinstance(other, Variable):
            other = other.value
        return self.value == other

    def backward(self):
        next = self.next
        if next:
            return self.next.backward() * self.grad
        else:
            return self.grad

    def __add__(self, other: Union[Number, "Variable"]) -> "Variable":
        if isinstance(other, Number):
            self.next = Variable(self.value + other)
        else:
            self.next = Variable(self.value + other.value)
        return self.next

    def __neg__(self) -> "Variable":
        self.value *= -1
        self.grad *= -1
        return self


if __name__ == "__main__":
    # y = a + 5
    a = Variable(3)
    y = a + 5
    assert y == 8
    assert a.backward() == 1

    # y = a + b + 5
    a = Variable(3)
    b = Variable(3)
    y = a + b + 5
    assert y == 11
    assert a.backward() == 1
    assert b.backward() == 1

    # y = -a
    a = Variable(3)
    y = -a
    assert y == -3
    assert a.backward() == -1
