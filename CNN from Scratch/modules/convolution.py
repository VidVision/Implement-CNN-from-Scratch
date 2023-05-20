#  Ref:
# https://numpy.org/doc/stable/reference/generated/numpy.pad.html
# https://www.deeplearningbook.org/contents/convnets.html
# https://medium.com/@pavisj/convolutions-and-backpropagations-46026a8f5d2c#6042
# https://numpy.org/doc/stable/reference/generated/numpy.zeros_like.html
"""
2d Convolution Module.
"""

import numpy as np


class Conv2D:
    '''
    An implementation of the convolutional layer. We convolve the input with out_channels different filters
    and each filter spans all channels in the input.
    '''

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0):
        """
        :param in_channels: the number of channels of the input data
        :param out_channels: the number of channels of the output(aka the number of filters applied in the layer)
        :param kernel_size: the specified size of the kernel(both height and width)
        :param stride: the stride of convolution
        :param padding: the size of padding. Pad zeros to the input with padding size.
        """
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.cache = None

        self._init_weights()

    def _init_weights(self):
        np.random.seed(1024)
        self.weight = 1e-3 * np.random.randn(self.out_channels, self.in_channels, self.kernel_size, self.kernel_size)
        self.bias = np.zeros(self.out_channels)

        self.dx = None
        self.dw = None
        self.db = None

    def forward(self, x):
        """
        The forward pass of convolution
        :param x: input data of shape (N, C, H, W)
        :return: output data of shape (N, self.out_channels, H', W') where H' and W' are determined by the convolution
                 parameters. Save necessary variables in self.cache for backward pass
        """
        out = None
        #############################################################################
        #       Implement the convolution forward pass.                             #
        # Hint: 1) You may use np.pad for padding.                                  #
        #       2) You may implement the convolution with loops                     #
        #############################################################################
        N, C, H, W = x.shape   #input data shape

        #calculate the output size
        H_prime = int((H + 2 * self.padding - self.kernel_size) / self.stride) + 1
        W_prime = int((W + 2 * self.padding - self.kernel_size) / self.stride) + 1

        # pad H and W dimensions in x(N,C,H,W)
        # https://numpy.org/doc/stable/reference/generated/numpy.pad.html
        x_padded = np.pad(x, ((0,0),
                              (0,0),
                              (self.padding,self.padding),
                              (self.padding,self.padding)),
                          'constant',
                          constant_values = 0)

        # initialize output populating by zero:
        out = np.zeros((N, self.out_channels, H_prime, W_prime))

        #perform convolution
        # https://www.deeplearningbook.org/contents/convnets.html
        for i in range(N):  #looping over number of samples
            for j in range(self.out_channels):  #looping over filters
                for a in range(H_prime):        # looping over output vertically
                    for b in range(W_prime):    # looping over output horizontally
                        h_start = a*self.stride  #vertical corners of window (slice that filter will be applied to)
                        h_end = h_start+self.kernel_size
                        w_start = b*self.stride  #horizontal corners of window (slice that filter will be applied to)
                        w_end = w_start+self.kernel_size
                        # sum of elementwise product of slice and weight(of each filter),  and sum over all entries
                        conv = np.sum((x_padded[i, :,h_start:h_end, w_start:w_end ])*(self.weight[j,:,:,:]))
                        # add bias for that filter and store in output
                        out[i,j,a,b] = conv + self.bias[j]

        self.cache = x
        return out

    def backward(self, dout):
        """
        The backward pass of convolution
        :param dout: upstream gradients
        :return: nothing but dx, dw, and db of self should be updated
        """
        x = self.cache
        #############################################################################
        # TODO: Implement the convolution backward pass.                            #
        # Hint:                                                                     #
        #       1) You may implement the convolution with loops                     #
        #       2) don't forget padding when computing dx                           #
        #############################################################################
        N, C, H, W = x.shape   #input data shape
        # pad H and W dimensions in x(N,C,H,W)
        # https://numpy.org/doc/stable/reference/generated/numpy.pad.html
        x_padded = np.pad(x, ((0,0),
                              (0,0),
                              (self.padding,self.padding),
                              (self.padding,self.padding)),
                          'constant',
                          constant_values = 0)
        # https://numpy.org/doc/stable/reference/generated/numpy.zeros_like.html
        # initialize padded dx wih zeros (same dim as padded x)
        dx_padded = np.zeros_like(x_padded)
        self.dx = np.zeros_like(x)
        # print('dx shape :',self.dx.shape)
        self.dw = np.zeros_like(self.weight)
        # print('dw shape :',self.dw.shape)
        self.db = dout.sum(axis=(0, 2, 3))  # equivalent to dot product of dout with ones tensor
        # print('db shape :',self.db.shape)

        #calculate the output size
        H_prime = int((H + 2 * self.padding - self.kernel_size) / self.stride) + 1
        W_prime = int((W + 2 * self.padding - self.kernel_size) / self.stride) + 1

        #perform convolution for backward:
        # https://medium.com/@pavisj/convolutions-and-backpropagations-46026a8f5d2c#6042
        # https://datascience-enthusiast.com/DL/dlindex.html
        # dw = conv (x, dout)
        # dx = full conv (rotated F, dout)

        for i in range(N):  #looping over number of samples
            for j in range(self.out_channels):  #looping over filters
                for a in range(H_prime):        # looping over output vertically
                    for b in range(W_prime):    # looping over output horizontally
                        h_start = a*self.stride  #vertical corners of window (slice that filter will be applied to)
                        h_end = h_start+self.kernel_size
                        w_start = b*self.stride  #horizontal corners of window (slice that filter will be applied to)
                        w_end = w_start+self.kernel_size
                        # Update the gradients
                        self.dw[j, :, :, :] += x_padded[i, :,h_start:h_end, w_start:w_end]*dout[i, j, a, b]
                        dx_padded[i, :, h_start:h_end, w_start:w_end] += self.weight[j,:,:,:] * dout[i, j, a, b]

        #remove padding
        self.dx = dx_padded[:, :, self.padding:(self.padding+H), self.padding:(self.padding+W)]
