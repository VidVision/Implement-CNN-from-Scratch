"""
Linear Module.
"""
import numpy as np


class Linear:
    """
    A linear layer with weight W and bias b. Output is computed by y = Wx + b
    """

    def __init__(self, in_dim, out_dim):
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.cache = None

        self._init_weights()

    def _init_weights(self):
        np.random.seed(1024)
        self.weight = 1e-3 * np.random.randn(self.in_dim, self.out_dim)
        np.random.seed(1024)
        self.bias = np.zeros(self.out_dim)

        self.dx = None
        self.dw = None
        self.db = None

    def forward(self, x):
        """
        Forward pass of linear layer
        :param x: input data, (N, d1, d2, ..., dn) where the product of d1, d2, ..., dn is equal to self.in_dim
        :return: The output computed by Wx+b. Save necessary variables in cache for backward
        """
        out = None
        #############################################################################
        # TODO: Implement the forward pass.                                         #
        #    HINT: You may want to flatten the input first                          #
        #############################################################################
        flat_x = np.reshape(x, (x.shape[0],-1))  # (N, in_dim)
        W = self.weight                          # (in_dim,out_dim)
        b = self.bias                            # (out_dim)
        # flatten x (1,in_dim)
        out = flat_x@W+b  #(N,in_dim).(in_dim,out_dim) = (N,out_dim)

        self.cache = x
        return out

    def backward(self, dout):
        """
        Computes the backward pass of linear layer
        :param dout: Upstream gradients, (N, self.out_dim)
        :return: nothing but dx, dw, and db of self should be updated
        """
        x = self.cache
        #############################################################################
        # TODO: Implement the linear backward pass.                                 #
        #############################################################################
        x = self.cache                           #(N,d1,d2,...)
        W = self.weight                          # (in_dim,out_dim)
        dx = dout@ W.T                           # (N,out_dim)(out_dim,in_dim) = (N,in_dim)
        input_shape = x.shape
        self.dx = np.reshape(dx, input_shape)  # (N,d1,d2,..)
        flat_x = np.reshape(x, (x.shape[0], -1))  # (N, in_dim)
        # print('flat_x shape :', flat_x.shape)
        self.dw = np.transpose(flat_x)@dout           # (in_dim,N)(N,out_dim) = (in_dim,out_dim)
        # print('dw shape :', self.dw.shape)
        self.db = dout.T@np.ones((x.shape[0],))       # (out_dim,N)(N,) = out_dim

