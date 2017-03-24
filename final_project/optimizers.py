# Imports

import numpy as np

# Classes


class SGD(object):
    """ Stochastic gradient descent optimizer

    Very stupid simple optimizer

    :param float learn_rate:
        The learning rate > 0
    :param float weight_decay:
        The weight_decay parameter > 0
    """

    def __init__(self, learn_rate=0.001, weight_decay=0.0):
        self.learn_rate = learn_rate
        self.weight_decay = weight_decay

    def update(self, weights, grads):
        """ Calculate the update for each layer

        :param weights:
            The list of weight/bias elements for each layer
        :param grads:
            The list of gradients for those corresponding layers
        :returns:
            The updated list of weights
        """
        learn_rate_t = self.learn_rate

        new_weights = []
        for w, g in zip(weights, grads):

            # Apply weight decay
            if w.shape[1] > 1:
                # Weight matrix
                g_t = learn_rate_t * (g + self.weight_decay * w)
            else:
                # Bias matrix
                g_t = learn_rate_t * g
            w_t = w - g_t
            new_weights.append(w_t)
        return new_weights


class Adam(object):
    """ Adam optimizer

    `Adam - A Method for Stochastic Optimization <http://arxiv.org/abs/1412.6980v8>`_

    Based on the implementation in
    `Keras <https://github.com/fchollet/keras/blob/master/keras/optimizers.py#L353>`_

    :param float learn_rate:
        The learning rate > 0
    :param float beta1:
        Magic momentum parameter #1 (0 < beta1 < 1)
    :param float beta2:
        Magic momentum parameter #2 (0 < beta2 < 1)
    """

    def __init__(self,
                 learn_rate=0.001,
                 beta1=0.9,
                 beta2=0.999):
        self.learn_rate = learn_rate
        self.beta1 = beta1
        self.beta2 = beta2

        self.iters = 0
        self.ms = None
        self.vs = None

    def update(self, weights, grads):
        """ Calculate the update for each layer

        :param weights:
            The list of weight/bias elements for each layer
        :param grads:
            The list of gradients for those corresponding layers
        :returns:
            The updated list of weights
        """

        learn_rate = self.learn_rate

        # Scale the learning rate by the iteration number
        t = self.iters + 1
        learn_rate_t = learn_rate * (np.sqrt(1 - self.beta2**t) /
                                     (1 - self.beta1**t))

        # Store the momentum and velocities for each node
        if self.ms is None:
            self.ms = [np.zeros(w.shape) for w in weights]
        if self.vs is None:
            self.vs = [np.zeros(w.shape) for w in weights]
        ms, vs = self.ms, self.vs

        # Make sure everything has the right length
        assert len(weights) == len(grads)
        assert len(weights) == len(ms)
        assert len(weights) == len(vs)

        # Now, for each weight stack, update momentum, velocity, weights
        new_ms = []
        new_vs = []
        new_weights = []
        for w, g, m, v in zip(weights, grads, ms, vs):
            # Momentum update
            m_t = (self.beta1 * m) + (1.0 - self.beta1) * g

            # Velocity update
            v_t = (self.beta2 * v) + (1.0 - self.beta2) * g**2

            # Update the weights
            w_t = w - learn_rate_t * m_t / (np.sqrt(v_t) + 1e-8)

            new_ms.append(m_t)
            new_vs.append(v_t)
            new_weights.append(w_t)

        self.ms = new_ms
        self.vs = new_vs
        return new_weights
