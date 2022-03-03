from typing import Optional


class tensor:

    def __init__(self, value, operator: Optional["Add"] = None):
        self.value = value
        self.operator = operator
        self.grad = 0

    def __repr__(self):
        return f"tensor({self.value}, grad_fn={self.operator})"

    def __add__(self, other):
        add = Add(self, other)
        next = tensor(add._forward(), add)
        return next

    def backward(self, grad=1):
        self.grad += grad
        if self.operator:
            self.operator._backward(self.grad)


class Add:

    def __init__(self, a, b):
        self.a = a if isinstance(a, tensor) else tensor(a)
        self.b = b if isinstance(b, tensor) else tensor(b)

    def __repr__(self):
        return f"<Add at 0x{id(self)}>"

    def _forward(self):
        return self.a.value + self.b.value

    def _backward(self, grad):
        self.a.backward(grad)
        self.b.backward(grad)


if __name__ == "__main__":
    a = tensor(5)
    y = a + a + a
    y.backward()
    print(a)
    print(y)
    print(a.grad)
