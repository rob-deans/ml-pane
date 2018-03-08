import numpy as np
from abc import ABCMeta, abstractmethod
import math


class Error:
    __metaclass__ = ABCMeta

    @staticmethod
    @abstractmethod
    def error(x, y):
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def derivative(x, y):
        raise NotImplementedError


class Absolute(Error):

    @staticmethod
    def error(x, y):
        return x - y

    @staticmethod
    def derivative(x, y):
        raise NotImplementedError


class MSE(Error):

    @staticmethod
    def error(actual, predicted):
        return 0.5 * np.mean(np.square(np.subtract(actual, predicted)))

    @staticmethod
    def derivative(actual, predicted):
        return actual - predicted


class RMSE(Error):

    @staticmethod
    def error(x, y):
        n = len(x)
        absolute = x - y
        squared = absolute**2
        summed = np.sum(squared)
        return np.sqrt(summed / n)

    @staticmethod
    def derivative(x, y):
        raise NotImplementedError


class Optimiser:
    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self, cost_function):
        self.cost_function = cost_function
        self.k = np.array([])
        self.layers = None
        self.learning_rate = 0.1
        self.epoch = 0

    @abstractmethod
    def optimise(self, actual):
        raise NotImplementedError


class GradientDescent(Optimiser):
    def __init__(self, cost_function):
        super(GradientDescent, self).__init__(cost_function)

    def optimise(self, actual):
        # Calculate the error for the output
        cost_derivative = self.cost_function.derivative(actual, self.k[-1])
        # error = self.cost_function.error(actual, self.k[-1])

        # mean_error = np.mean(np.abs(cost_derivative))

        # Calculate the delta for the final layer
        output_delta = cost_derivative * self.layers[-1].activation.derivative(self.k[-1])

        deltas = [output_delta]

        for l in range(len(self.layers) - 1, 0, -1):
            layer = self.layers[l]
            error = deltas[-1].dot(layer.weights.T)
            deltas.append(error * layer.activation.derivative(self.k[l]))

        deltas = list(reversed(deltas))

        for l, layer in enumerate(self.layers):
            layer.update(self.k[l].T.dot(deltas[l]) * self.learning_rate)

        return cost_derivative


class Annealing:
    epoch = 0

    def __init__(self, start, end, epochs):
        self.start = start
        self.end = end
        self.epochs = epochs

    def anneal(self, epoch):
        exponent = (20 * epoch) / self.epochs
        exp = math.exp(10 - exponent)
        temp = 1 - (1 / (1 + exp))
        return self.end + ((self.start - self.end) * temp)

