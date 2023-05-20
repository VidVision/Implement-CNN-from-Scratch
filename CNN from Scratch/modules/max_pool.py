# Ref:
# https://stanford.edu/~shervine/teaching/cs-230/cheatsheet-convolutional-neural-networks
"""
2d Max Pooling Module.
"""

import numpy as np


class MaxPooling:
    """
    Max Pooling of input
    """

    def __init__(self, kernel_size, stride):
        self.kernel_size = kernel_size
        self.stride = stride
        self.cache = None
        self.dx = None

    def forward(self, x):
        """
        Forward pass of max pooling
        :param x: input, (N, C, H, W)
        :return: The output by max pooling with kernel_size and stride
        """
        out = None
        #############################################################################
        # TODO: Implement the max pooling forward pass.                             #
        # Hint:                                                                     #
        #       1) You may implement the process with loops                         #
        #############################################################################
        N, C, H, W = x.shape  # input data shape

        # calculate the output size
        H_out = int((H - self.kernel_size) / self.stride) + 1
        W_out = int((W - self.kernel_size) / self.stride) + 1

        # initialize output:
        out = np.zeros((N, C, H_out, W_out))

        # perform max pooling
        for i in range(N):  # looping over number of samples
            for j in range(C):  # looping over channels
                for a in range(H_out):  # looping over output vertically
                    for b in range(W_out):  # looping over output horizontally
                        h_start = a * self.stride  # vertical corners of window
                        h_end = h_start + self.kernel_size
                        w_start = b * self.stride  # horizontal corners of window
                        w_end = w_start + self.kernel_size
                        # max of the window
                        out[i, j, a, b] = np.max(x[i, j, h_start:h_end, w_start:w_end])

        self.cache = (x, H_out, W_out)
        return out

    def backward(self, dout):
        """
        Backward pass of max pooling
        :param dout: Upstream derivatives
        :return:
        """
        x, H_out, W_out = self.cache
        #############################################################################
        # TODO: Implement the max pooling backward pass.                            #
        # Hint:                                                                     #
        #       1) You may implement the process with loops                     #
        #       2) You may find np.unravel_index useful                             #
        #############################################################################
        N, C, H, W = x.shape  # input data shape - # of data records, # of input channels, input height, input width
        # initialize dx , same dim as x:
        self.dx = np.zeros_like(x)

        # perform max pooling backward
        # Retrieve the indices of the maximum values from the forward pass.
        for i in range(N):  # looping over number of samples
            for j in range(C):  # looping over channels
                for a in range(H_out):  # looping over output vertically
                    for b in range(W_out):  # looping over output horizontally
                        h_start = a * self.stride  # vertical corners of window
                        h_end = h_start + self.kernel_size
                        w_start = b * self.stride  # horizontal corners of window
                        w_end = w_start + self.kernel_size
                        # get index of max in the window
                        ind = np.argmax(x[i, j, h_start:h_end, w_start:w_end])
                        # https://stackoverflow.com/questions/48135736/what-is-an-intuitive-explanation-of-np-unravel-index/48136499
                        h_idx, w_idx = np.unravel_index(ind, (self.kernel_size, self.kernel_size))
                        # For each maximum value index, assign the corresponding error gradient from the next layer to the corresponding
                        # location in the gradient map.
                        self.dx[i, j, h_start:h_end, w_start:w_end][h_idx, w_idx] = dout[i, j, a, b]

