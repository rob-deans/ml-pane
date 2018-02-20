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
    def derivative():
        raise NotImplementedError


class Adaline(Error):

    @staticmethod
    def error(x, y):
        return x - y

    @staticmethod
    def derivative():
        pass


class MSE(Error):

    @staticmethod
    def error(x, y):
        # n = len(x)
        # absolute = x - y
        # squared = absolute**2
        # summed = np.sum(squared)
        # mean = summed / n
        mean = np.square(np.subtract(x, y)).mean()
        return np.full_like(x, mean)

    @staticmethod
    def derivative():
        pass


class RMSE(Error):

    @staticmethod
    def error(x, y):
        n = len(x)
        absolute = x - y
        squared = absolute**2
        summed = np.sum(squared)
        return np.sqrt(summed / n)

    @staticmethod
    def derivative():
        pass

