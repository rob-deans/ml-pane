import numpy as np
from data_processing import get_data


def sigmoid(x, deriv=False):
    if deriv:
        return x * (1 - x)

    return 1 / (1 + np.exp(-x))


data, res = get_data()

np.random.seed(1)

# randomly initialize our weights with mean 0
syn0 = 2 * np.random.random((5, 8)) - 1
syn1 = 2 * np.random.random((8, 1)) - 1

w0 = 2 * np.random.random((3, 4)) - 1
w1 = 2 * np.random.random((4, 1)) - 1

for j in xrange(50000):

    # Feed forward through layers 0, 1, and 2
    k0 = data
    k1 = sigmoid(np.dot(k0, syn0))
    k2 = sigmoid(np.dot(k1, syn1))

    # how much did we miss the target value?
    k2_error = res - k2

    if (j % 10000) == 0:
        print "Error:" + str(np.mean(np.abs(k2_error)))

    # in what direction is the target value?
    # were we really sure? if so, don't change too much.
    k2_delta = k2_error * sigmoid(k2, deriv=True)

    # how much did each k1 value contribute to the k2 error (according to the weights)?
    k1_error = k2_delta.dot(syn1.T)

    # in what direction is the target k1?
    # were we really sure? if so, don't change too much.
    k1_delta = k1_error * sigmoid(k1, deriv=True)

    syn1 += k1.T.dot(k2_delta * 0.1)
    syn0 += k0.T.dot(k1_delta * 0.1)

for i, d in enumerate(data[-10:]):
    k0 = d
    k1 = sigmoid(np.dot(k0, syn0))
    k2 = sigmoid(np.dot(k1, syn1))
    result = sigmoid(k2, deriv=True)
    error = result - res[-10:][i]
    print('Error: {}'.format(error))

