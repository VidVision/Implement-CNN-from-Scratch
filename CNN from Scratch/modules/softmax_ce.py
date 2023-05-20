"""
Softmax Cross Entropy Module.
"""
import numpy as np


class SoftmaxCrossEntropy:
    """
    Compute softmax cross-entropy loss given the raw scores from the network.
    """

    def __init__(self):
        self.dx = None
        self.cache = None

    def forward(self, x, y):
        """
        Compute Softmax Cross Entropy Loss
        :param x: raw output of the network: (N, num_classes)
        :param y: labels of samples: (N, )
        :return: computed CE loss of the batch
        """
        # By subtracting the maximum value from each element of the input vector, we ensure that the largest exponentiated
        # value is zero, and all other values are negative or zero. This adjustment helps to prevent overflow issues.
        probs = np.exp(x - np.max(x, axis=1, keepdims=True))
        probs /= np.sum(probs, axis=1, keepdims=True)
        N, _ = x.shape
        loss = -np.sum(np.log(probs[np.arange(N), y])) / N
        self.cache = (probs, y, N)
        return probs, loss

    def backward(self):
        """
        Compute backward pass of the loss function
        :return: gradient of the loss with respect to the input
        """
        # The gradients are computed by subtracting 1 from the corresponding class probabilities and dividing by the
        # batch size N.
        probs, y, N = self.cache
        dx = probs.copy()
        # substracting 1 ensures that when we compute the gradient, the corresponding term becomes (e^x - 1) / (e^x + ... + e^x) instead
        # of e^x / (e^x + ... + e^x). This modification simplifies the derivative calculation and helps propagate the gradients correctly.
        dx[np.arange(N), y] -= 1  # dx np.arange(N), y] selects the probabilities corresponding to the correct class labels
        # divide by N to normalize the gradients and obtain the average gradient per sample in the batch.
        dx /= N
        self.dx = dx
