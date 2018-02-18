import numpy as np
from data_processing import get_all_data
from data_processing import denormalise
from activation import Sigmoid
from activation import Linear
from activation import Relu


class NoDataException(Exception):
    pass


class NotLayerException(Exception):
    pass


data, res, max_, min_ = get_all_data()

np.random.seed(1)


class Layer:
    def __init__(self, inputs, units, activation, initialiser_fn='default'):
        self.inputs = inputs
        self.units = units
        self.activation = activation
        self.initialiser_fn = initialiser_fn

        self.weights = 2 * np.random.random((self.inputs, self.units)) - 1

    def ff(self, values):
        return self.activation.activation_fn(np.dot(values, self.weights))

    def delta(self, actual):
        pass

    def update_weights(self, delta):
        self.weights += delta


class Network:
    def __init__(self, learning_rate=0.1, layers=None):
        self.k = []
        self.data = None
        self.learning_rate = learning_rate
        if layers is None:
            raise NotLayerException
        self.layers = layers
        # print(type(self.layers[0]))
        # if any(isinstance(x, Layer) for x in self.layers):
        #     raise NotLayerException

    def run(self, values):
        self.data = values
        del self.k[:]
        self.k.append(self.data)
        for layer in self.layers:
            self.k.append(layer.ff(self.k[-1]))
        return self.k[-1]

    def optimise(self, actual):

        if self.data is None or self.k == []:
            raise NoDataException

        # Calculate the error for the output
        kl_error = actual - self.k[-1]

        mean_error = np.mean(np.abs(kl_error))

        # Calculate the delta for the final layer
        kl_delta = kl_error * self.layers[-1].activation.derivative(self.k[-1])

        deltas = [kl_delta]

        for l in range(len(self.layers) - 1, 0, -1):
            layer = self.layers[l]
            errors = deltas[-1].dot(layer.weights.T)
            deltas.append(errors * layer.activation.derivative(self.k[l]))

        for i, layer in enumerate(self.layers):
            layer.update_weights(delta=self.k[i].T.dot(deltas[-(i + 1)] * self.learning_rate))

        return mean_error

    def save(self):
        pass

    def load(self):
        pass


layer1 = Layer(inputs=5, units=8, activation=Sigmoid)
layer2 = Layer(inputs=8, units=8, activation=Sigmoid)
layer3 = Layer(inputs=8, units=8, activation=Sigmoid)
layer4 = Layer(inputs=8, units=1, activation=Linear)

network = Network(learning_rate=0.0001, layers=[layer1, layer2, layer3, layer4])

for j in xrange(1000000):

    # Feed forward through layers 0, 1, and 2
    k2 = network.run(data)
    _error = network.optimise(res)
    if (j % 100000) == 0:
        print "Error:" + str(np.mean(np.abs(_error)))


for i, d in enumerate(data[-10:]):
    k0 = network.run(d)
    result_de = denormalise(k0, max_.values.tolist()[-1], min_.values.tolist()[-1])
    actual_de = denormalise(res[-10:][i], max_.values.tolist()[-1], min_.values.tolist()[-1])
    error = result_de - actual_de
    print('Predicted: {} | Actual: {} | Error: {}'.format(result_de, actual_de, np.abs(error)))

