from NN import MLP
from Value import Value


class XOR:
    def __init__(self, xs, ys, model):
        self.model = model
        self.xs = xs
        self.ys = ys
        # This holds the data predicted by the model
        self.y_pred = []
        self.loss = Value(0)

    ETA = -0.05

    def _forward(self):
        self.y_pred = []
        for x in xs:
            self.y_pred.append(self.model.forward(x))

    def train(self):
        for i in range(300):

            self._forward()

            # Calculate loss
            self.loss = 0
            for y1, y2 in zip(self.y_pred, self.ys):
                self.loss = self.loss + (y1[0] - y2) ** 2

            # clear the gradient
            for p in self.model.parameters():
                p.grad = 0

            # Paramaters
            print(f"Parameters : {len(self.model.parameters())}")

            # Back propagation
            self.loss.calculatebackward()

            # Now we update the weights, we keep the learning rate negative as we want to decrease the loss
            for p in self.model.parameters():
                p.data += (XOR.ETA * p.grad)

            for y in self.y_pred:
                print(f"Predicted data : {round(y[0].data, 3)} for {i} iteration(s)")
            print(f"Actual loss : {self.loss.data}")

            if self.loss.data < 0.01:
                break


xs = [
    [1.0, 0],
    [1.0, 1.0],
    [0, 1.0],
    [0, 0]
]

ys = [1, 0, 1, 0]

# Initialize the model
model = MLP(2, [3, 3, 2, 4, 2, 4, 6, 1])

xor = XOR(xs, ys, model)
xor.train()