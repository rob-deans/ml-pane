import numpy as np
from abc import ABCMeta, abstractmethod


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


class Adaline(Error):

    @staticmethod
    def error(x, y):
        return x - y

    @staticmethod
    def derivative(x, y):
        pass


class SE(Error):

    @staticmethod
    def error(actual, predicted):
        return np.square(np.subtract(actual, predicted))

    @staticmethod
    def derivative(actual, predicted):
        return 2 * (predicted - actual)


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
        pass

