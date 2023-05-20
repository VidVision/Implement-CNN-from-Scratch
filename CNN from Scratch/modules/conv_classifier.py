"""
CovNet Module.
"""

from .softmax_ce import SoftmaxCrossEntropy
from .relu import ReLU
from .max_pool import MaxPooling
from .convolution import Conv2D
from .linear import Linear


class ConvNet:
    """
    Max Pooling of input
    """
    def __init__(self, modules, criterion):
        self.modules = []
        for m in modules:
            if m['type'] == 'Conv2D':
                self.modules.append(
                    Conv2D(m['in_channels'],
                           m['out_channels'],
                           m['kernel_size'],
                           m['stride'],
                           m['padding'])
                )
            elif m['type'] == 'ReLU':
                self.modules.append(
                    ReLU()
                )
            elif m['type'] == 'MaxPooling':
                self.modules.append(
                    MaxPooling(m['kernel_size'],
                               m['stride'])
                )
            elif m['type'] == 'Linear':
                self.modules.append(
                    Linear(m['in_dim'],
                           m['out_dim'])
                )
        if criterion['type'] == 'SoftmaxCrossEntropy':
            self.criterion = SoftmaxCrossEntropy()
        else:
            raise ValueError("Wrong Criterion Passed")

    def forward(self, x, y):
        """
        The forward pass of the model
        :param x: input data: (N, C, H, W)
        :param y: input label: (N, )
        :return:
          probs: the probabilities of all classes: (N, num_classes)
          loss: the cross entropy loss
        """
        probs = None
        loss = None
        #############################################################################
        #    1) Implement forward pass of the model                                 #
        #############################################################################
        layer_input = x
        # loop over different layers, the input to each layer is the output of previous layer
        for m in self.modules:
            layer_output = m.forward(layer_input)
            layer_input = layer_output
        # final output will be the input to Softmax_CE
        probs, loss = self.criterion.forward(layer_input, y)

        return probs, loss

    def backward(self):
        """
        The backward pass of the model
        :return: nothing but dx, dw, and db of all modules are updated
        """
        #############################################################################
        #    1) Implement backward pass of the model                                #
        #############################################################################
        # calculate dx - backward pass of softmax_CE to be the input of next layer (in reverse order)
        self.criterion.backward()
        layer_input = self.criterion.dx
        # go over the layers list from end to beginning
        for m in self.modules[::-1]:
            m.backward(layer_input)
            layer_input = m.dx
