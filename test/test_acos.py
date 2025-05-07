import numpy as np
from mindspore import Tensor, context
from mindspore.nn import Cell
from mindspore import mint

context.set_context(device_target="CPU", mode=1)


class Net(Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.op = mint.acos

    def construct(self, x):
        return self.op(x)


if __name__ == "__main__":
    x0 = Tensor(np.array([[0.0, -0.1], [-0.2, 1.0]]).astype(np.float32))
    net = Net()
    output = net(x0)
    print(output)