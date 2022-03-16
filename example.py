import numpy as np

from autograd import Tensor, exp

a = Tensor(3)

s = 1 / (1 + exp(-a))
s.backward()

assert np.isclose(s.value, 1 / (1 + np.exp(-3)))
assert np.isclose(a.grad, s.value * (1 - s.value))
