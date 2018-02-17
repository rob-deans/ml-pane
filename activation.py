import math as maths


def relu(i):
    return i if i > 0 else 0


def sigmoid(x):
    return 1 / (1 + maths.exp(-x))


def tanh():
    pass

