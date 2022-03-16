import warnings

import matplotlib.pyplot as plt
import numpy as np

from autograd import Tensor
from autograd.transformers import exp

warnings.simplefilter("ignore", RuntimeWarning)


def f(x):
    if isinstance(x, Tensor):
        return exp(-x**2)
    else:
        return np.exp(-x**2)


def grad_descent(start, eta=1, max_iter=30):
    t = Tensor(start)
    for _ in range(max_iter):
        yield t.value, f(t.value)

        f(t).backward()
        if np.abs(t.grad) < 1e-15:
            raise StopIteration
        else:
            t = t + Tensor(eta * t.grad)


x = np.linspace(-2, 2, 50)
x = Tensor(x)
y = f(x).backward()
opt_x, opt_y = zip(*grad_descent(start=2))

fig, axes = plt.subplots(1, 2, squeeze=True)
for ax in axes:
    ax.plot(x.value, y.value)
    ax.plot(x.value, x.grad)
    ax.plot(opt_x, opt_y, "ro-")
    ax.grid(True)
ax.set(xlim=[-0.3, 0.3], ylim=[0.8, None])
plt.show()
