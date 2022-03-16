from numbers import Number
from typing import Iterable, Optional, Union

import numpy as np


def asymmetric(method):

    def swap(self, other) -> "Tensor":
        return method(other, self)

    return swap


class Tensor:

    def __init__(self, value: Union[Number, Iterable],
                 operator: Optional["Operator"] = None):
        self.value = np.array(value, dtype=float)
        self.operator = operator
        self.grad = 0

    def __repr__(self) -> str:
        return f"Tensor({self.value}, grad_fn={self.operator})"

    def backward(self, grad: Number = None) -> "Tensor":
        if grad is None:
            grad = np.ones_like(self.value)
        self.grad += grad
        if self.operator:
            self.operator.calc_backward(self.grad)
        return self

    def __neg__(self) -> "Tensor":
        """符号反転"""
        return -1 * self

    def __add__(self, other) -> "Tensor":
        """足し算"""
        add = Add(self, other)
        next = Tensor(add.calc_forward(), add)
        return next

    def __sub__(self, other) -> "Tensor":
        """引き算 (self - other)"""
        return self + (-other)

    def __mul__(self, other) -> "Tensor":
        """掛け算"""
        mul = Mul(self, other)
        next = Tensor(mul.calc_forward(), mul)
        return next

    def __truediv__(self, other) -> "Tensor":
        """割り算 (self / other)"""
        return self * other**-1

    def __pow__(self, other) -> "Tensor":
        """べき乗 (self ** other)"""
        pow = Pow(self, other)
        next = Tensor(pow.calc_forward(), pow)
        return next

    __radd__ = __add__
    __rsub__ = asymmetric(__sub__)
    __rmul__ = __mul__
    __rtruediv__ = asymmetric(__truediv__)
    __rpow__ = asymmetric(__pow__)


from .operators import Add, Mul, Operator, Pow  # noqa
