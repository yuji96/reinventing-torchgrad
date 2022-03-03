import math

from autograd import Tensor, exp

a = Tensor(3)

s = 1 / (1 + exp(-a))
s.backward()

assert math.isclose(s.value, 1 / (1 + math.exp(-3)))
assert math.isclose(a.grad, s.value * (1 - s.value))
