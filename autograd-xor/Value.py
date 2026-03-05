import math


class Value:
    def __init__(self, data, children=(), op=''):
        self.data = data
        self.children = set(children)
        self.op = op
        self.backward = lambda: None
        self.grad = 0

    def __repr__(self):
        return f"Value : ({self.data})"

    def __add__(self, arg):
        arg = arg if isinstance(arg, Value) else Value(arg)
        out = Value(self.data + arg.data, (self, arg), '+')

        def _backward():
            self.grad += 1 * out.grad
            arg.grad += 1 * out.grad

        out.backward = _backward
        return out

    def __mul__(self, arg):
        arg = arg if isinstance(arg, Value) else Value(arg)
        out = Value(self.data * arg.data, (self, arg), '*')

        def _backward():
            self.grad += arg.data * out.grad
            arg.grad += self.data * out.grad

        out.backward = _backward
        return out

    def __rmul__(self, other):  # other * self
        return self * other

    def __radd__(self, other):  # other + self
        return self + other

    def tanh(self):
        x = self.data
        t = (math.exp(2 * x) - 1) / (math.exp(2 * x) + 1)
        out = Value(t, (self,), 'tanh')

        def _backward():
            self.grad += (1 - t ** 2) * out.grad

        out.backward = _backward
        return out

    def __sub__(self, other):  # self - other
        return self + (-other)

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only supporting int/float powers for now"
        out = Value(self.data ** other, (self,), f'**{other}')

        def _backward():
            self.grad += other * (self.data ** (other - 1)) * out.grad

        out.backward = _backward
        return out

    def __neg__(self):
        return self * -1

    def sigmoid(self):
        s = 1 / (1 + math.exp(-self.data))
        out = Value(s, (self,), 'sigmoid')

        def _backward():
            out.grad += s * (1 - s) * out.grad

        out.backward = _backward
        return out

    def calculatebackward(self):

        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v.children:
                    build_topo(child)
                topo.append(v)

        build_topo(self)

        self.grad = 1

        for node in reversed(topo):
            node.backward()
