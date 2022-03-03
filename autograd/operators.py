from .tensor import Operator


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



