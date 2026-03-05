import random

from Value import Value


class Neuron:
    def __init__(self, inputs):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(inputs)]
        self.b = Value(random.uniform(-1, 1))

    def forward(self, x):
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        out = act.tanh()
        return out

    def parameters(self):
        return self.w + [self.b]


class Layer:
    def __init__(self, inp, out):
        self.n = [Neuron(inp) for _ in range(out)]

    def forward(self, x):
        outputs = []
        for n in self.n:
            outputs.append(n.forward(x))
        return outputs

    def parameters(self):
        params = []
        for nn in self.n:
            params.extend(nn.parameters())
        return params


class MLP:
    def __init__(self, inp, non):
        sizes = [inp] + non
        self.listOfLayers = []
        for i in range(len(non)):
            self.listOfLayers.append(Layer(sizes[i], sizes[i + 1]))

    def forward(self, x):
        for l in self.listOfLayers:
            x = l.forward(x)
        return x

    def parameters(self):
        params = []
        for nn in self.listOfLayers:
            params.extend(nn.parameters())
        return params
