#Ref:
# https://machinelearningmastery.com/rectified-linear-activation-function-for-deep-learning-neural-networks/
# https://ml-cheatsheet.readthedocs.io/en/latest/activation_functions.html#relu


"""
ReLU Module.
"""

import numpy as np


class ReLU:
    """
    An implementation of rectified linear units(ReLU)
    """

    def __init__(self):
        self.cache = None
        self.dx = None

    def forward(self, x):
        '''
        The forward pass of ReLU. Save necessary variables for backward
        :param x: input data
        :return: output of the ReLU function
        '''
        out = None
        #############################################################################
        # TODO: Implement the ReLU forward pass.                                    #
        #############################################################################
        # https://machinelearningmastery.com/rectified-linear-activation-function-for-deep-learning-neural-networks/
        out = np.maximum(0.0, x)
        self.cache = x
        return out

    def backward(self, dout):
        """
        :param dout: the upstream gradients
        :return:
        """
        dx, x = None, self.cache
        #############################################################################
        # TODO: Implement the ReLU backward pass.                                   #
        #############################################################################
        # https://ml-cheatsheet.readthedocs.io/en/latest/activation_functions.html#relu
        dx = (x > 0.0)*dout  # elementwise multiplication , dout if >0 else 0
        self.dx = dx
