from numbers import Number
from typing import Optional


class Tensor:

    def __init__(self, value: Number, operator: Optional["Operator"] = None):
        self.value = value
        self.operator = operator
        self.grad = 0

    def __repr__(self) -> str:
        return f"Tensor({self.value}, grad_fn={self.operator})"

    def backward(self, grad: Number = 1) -> None:
        self.grad += grad
        if self.operator:
            self.operator.calc_backward(self.grad)

    def __add__(self, other) -> "Tensor":
        add = Add(self, other)
        next = Tensor(add.calc_forward(), add)
        return next

    def __mul__(self, other) -> "Tensor":
        mul = Mul(self, other)
        next = Tensor(mul.calc_forward(), mul)
        return next

    def __pow__(self, other) -> "Tensor":
        pow = Pow(self, other)
        next = Tensor(pow.calc_forward(), pow)
        return next


from .operators import Add, Mul, Operator, Pow  # noqa
