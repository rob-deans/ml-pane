import numpy as np
from abc import ABCMeta, abstractmethod


class Activation:
    __metaclass__ = ABCMeta

    @staticmethod
    @abstractmethod
    def activation_fn(x):
        pass

    @staticmethod
    @abstractmethod
    def derivative(x):
        pass


class Sigmoid(Activation):
    def __init__(self):
        pass

    @staticmethod
    def activation_fn(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def derivative(x):
        return x * (1 - x)


class Tanh(Activation):
    def __init__(self):
        pass

    @staticmethod
    def activation_fn(x):
        return np.tanh(x)

    @staticmethod
    def derivative(x):
        return np.arctanh(x)


class Relu(Activation):
    def __init__(self):
        pass

    @staticmethod
    def activation_fn(x):
        return x if x > 0 else 0

    @staticmethod
    def derivative(x):
        return 1 if x > 0 else 0


class Linear(Activation):
    def __init__(self):
        pass

    @staticmethod
    def activation_fn(x):
        return x

    @staticmethod
    def derivative(x):
        return 1

