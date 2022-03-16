from numbers import Number

import numpy as np

from .tensor import Tensor


class Transformer:

    def __init__(self, a: Tensor) -> None:
        self.a = a

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}"

    def calc_forward(self) -> Number:
        raise NotImplementedError

    def calc_backward(self) -> None:
        raise NotImplementedError


class Log(Transformer):

    def calc_forward(self) -> Number:
        return np.log(self.a.value)

    def calc_backward(self, grad) -> None:
        self.a.backward(grad / self.a.value)


class ReLU(Transformer):

    def calc_forward(self) -> Number:
        return np.maximum(0, self.a.value)

    def calc_backward(self, grad: np.ndarray) -> None:
        self.a.backward(np.where(self.a.value > 0, grad, 0))


def log(a: Tensor) -> Tensor:
    _log = Log(a)
    return Tensor(_log.calc_forward(), _log)


def relu(a: Tensor) -> Tensor:
    _relu = ReLU(a)
    return Tensor(_relu.calc_forward(), _relu)


def exp(a: Tensor) -> Tensor:
    return np.e**a
