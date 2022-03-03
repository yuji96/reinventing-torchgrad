from numbers import Number
from typing import Optional


class Tensor:

    def __init__(self, value: Number, operator: Optional["Operator"] = None):
        self.value = value
        self.operator = operator
        self.grad = 0

    def __repr__(self) -> str:
        return f"Tensor({self.value}, grad_fn={self.operator})"

    def backward(self, grad: Number = 1) -> "Tensor":
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

    __radd__ = __add__

    def __sub__(self, other) -> "Tensor":
        """引き算"""
        return -1 * other + self

    def __rsub__(other, self) -> "Tensor":
        return other - self

    def __mul__(self, other) -> "Tensor":
        """掛け算"""
        mul = Mul(self, other)
        next = Tensor(mul.calc_forward(), mul)
        return next

    __rmul__ = __mul__

    def __truediv__(self, other) -> "Tensor":
        """割り算"""
        return 1 / other * self

    def __rtruediv__(self, other) -> "Tensor":
        return other * self**-1

    def __pow__(self, other) -> "Tensor":
        """べき乗"""
        # TODO: 確認
        pow = Pow(self, other)
        next = Tensor(pow.calc_forward(), pow)
        return next

    def __rpow__(self, other) -> "Tensor":
        pow = Pow(other, self)
        next = Tensor(pow.calc_forward(), pow)
        return next


from .operators import Add, Mul, Operator, Pow  # noqa
