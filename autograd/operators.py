from numbers import Number

import numpy as np

from .tensor import Tensor


class Operator:

    def __init__(self, a, b) -> None:
        self.a = a if isinstance(a, Tensor) else Tensor(a)
        self.b = b if isinstance(b, Tensor) else Tensor(b)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}"

    @property
    def values(self):
        return self.a.value, self.b.value

    def calc_forward(self) -> Number:
        raise NotImplementedError

    def calc_backward(self) -> None:
        raise NotImplementedError


class Add(Operator):

    def calc_forward(self):
        return self.a.value + self.b.value

    def calc_backward(self, grad):
        self.a.backward(grad)
        self.b.backward(grad)


class Mul(Operator):

    def calc_forward(self):
        return self.a.value * self.b.value

    def calc_backward(self, grad):
        self.a.backward(grad * self.b.value)
        self.b.backward(grad * self.a.value)


class Pow(Operator):

    def calc_forward(self):
        return self.a.value**self.b.value

    def calc_backward(self, grad):
        a = self.a.value
        b = self.b.value
        self.a.backward(grad * b * a**(b - 1))
        self.b.backward(grad * np.log(a) * a**b)
