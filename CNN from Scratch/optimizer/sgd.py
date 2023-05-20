"""
SGD Optimizer.
"""

from ._base_optimizer import _BaseOptimizer


class SGD(_BaseOptimizer):
    def __init__(self, model, learning_rate=1e-4, reg=1e-3, momentum=0.9):
        super().__init__(model, learning_rate, reg)
        self.momentum = momentum

    def update(self, model):
        """
        Update model weights based on gradients
        :param model: The model to be updated
        :return: None, but the model weights should be updated
        """
        self.apply_regularization(model)

        for idx, m in enumerate(model.modules):
            if hasattr(m, 'weight'):
                #############################################################################
                # TODO:                                                                     #
                #    1) Momentum updates for weights                                        #
                #############################################################################
                v_w = self.grad_tracker[idx]['dw']
                beta = self.momentum
                etha = self.learning_rate
                dL_dw = m.dw

                self.grad_tracker[idx]['dw'] = beta * v_w - etha *dL_dw
                m.weight += self.grad_tracker[idx]['dw']

            if hasattr(m, 'bias'):
                #############################################################################
                # TODO:                                                                     #
                #    1) Momentum updates for bias                                           #
                #############################################################################
                v_b = self.grad_tracker[idx]['db']
                beta = self.momentum
                etha = self.learning_rate
                dL_db = m.db
                bias = m.bias

                self.grad_tracker[idx]['db'] = beta * v_b - etha * dL_db
                m.bias += self.grad_tracker[idx]['db']


