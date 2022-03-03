import math
from numbers import Number

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
        return math.log(self.a.value)

    def calc_backward(self, grad):
        self.a.backward(grad / self.a.value)


def log(a: Tensor) -> Tensor:
    _log = Log(a)
    return Tensor(_log.calc_forward(), _log)
