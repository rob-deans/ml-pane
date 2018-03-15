import numpy as np
from abc import ABCMeta, abstractmethod


class Activation:
    __metaclass__ = ABCMeta

    @staticmethod
    @abstractmethod
    def activation_fn(x):
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def derivative(x):
        raise NotImplementedError


class Sigmoid(Activation):
    def __init__(self):
        pass

    @staticmethod
    def activation_fn(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def derivative(x):
        return x * (1 - x)


class TanH(Activation):
    def __init__(self):
        pass

    @staticmethod
    def activation_fn(x):
        return np.tanh(x)

    @staticmethod
    def derivative(x):
        return 1 - np.tanh(x) ** 2


class ReLu(Activation):
    def __init__(self):
        pass

    @staticmethod
    def activation_fn(x):
        return np.maximum(np.zeros_like(x), x)

    @staticmethod
    def derivative(x):
        x[x < 0] = 0
        return x


class LeakyReLu(Activation):
    def __init__(self):
        pass

    @staticmethod
    def activation_fn(x):
        return np.maximum(np.zeros_like(x), x)

    @staticmethod
    def derivative(x):
        def get_map(y):
            return 1 if y > 0 else 0.01 * y
        v = np.vectorize(get_map)
        return v(x)


class Linear(Activation):
    def __init__(self):
        pass

    @staticmethod
    def activation_fn(x):
        return x

    @staticmethod
    def derivative(x):
        return 1

