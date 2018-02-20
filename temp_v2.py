import numpy as np
from data_processing import get_all_data
from data_processing import denormalise
from activation import Sigmoid
from activation import Linear
from error import *


class NoDataException(Exception):
    pass


class NotLayerException(Exception):
    pass


data, max_, min_ = get_all_data()
np.random.seed(1)
np.random.shuffle(data)

TRAINING_PERCENTAGE = 0.6
DATA_LENGTH = len(data)
TRAINING_SIZE = int(DATA_LENGTH * TRAINING_PERCENTAGE)

training = data[:TRAINING_SIZE]
test = data[TRAINING_SIZE:]

# training = data[:1000]
# test = data[1000:]

training_data = training[:, :-1]
training_res = training[:, -1:]
test_data = test[:, :-1]
test_res = test[:, -1:]


class Layer:
    def __init__(self, inputs, units, activation, name=None):
        self.inputs = inputs
        self.units = units
        self.activation = activation
        self.name = name
        if self.name is None:
            self.name = 'layer_{}_{}'.format(self.inputs, self.units)

        self.weights = np.random.normal(0, 0.01, (self.inputs, self.units))
        self.biases = np.random.normal(0, size=self.units)

    def ff(self, values):
        return self.activation.activation_fn(np.dot(values, self.weights) + self.biases)

    def update(self, delta, bias_delta):
        self.weights -= delta
        self.biases -= bias_delta


class Network:
    def __init__(self, learning_rate=0.1, layers=None, cost_function=None):
        self.k = []
        self.data = None
        self.learning_rate = learning_rate
        if layers is None:
            raise Exception('No layers specified')
        self.layers = layers
        self.cost_function = cost_function

        for l in range(1, len(self.layers)):
            if self.layers[l].inputs != self.layers[l-1].units:
                raise Exception('{} does not have the same number of inputs ({}) as {} has units ({})'.format(
                    self.layers[l].name, self.layers[l].inputs, self.layers[l-1].name, self.layers[l-1].units
                ))

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

        predicted = self.k[-1]
        # Calculate the error for the output
        # kl_error = self.cost_function.error(actual, predicted)

        # Error for printing out during training
        absolute_error = self.cost_function.error(actual, predicted)

        cost_derive_1 = self.cost_function.derivative(actual, predicted)
        activation_derive_1 = self.layers[-1].activation.derivative(predicted)

        # weights/bias change for the last layer
        accumulative_derive = cost_derive_1 * activation_derive_1 * self.k[1]

        # convert to the correct shape for the multiplying
        weight_change = np.array([[d] for d in accumulative_derive])
        bias_change = cost_derive_1 * activation_derive_1

        self.layers[-1].update(weight_change * self.learning_rate, bias_change * self.learning_rate)

        # activation_derive_2 = self.layers[0].activation.derivative(self.k[1])
        # accumulative_derive *= activation_derive_2

        # weight_change_2 = []
        # for i in range(5):
        #     holder = []
        #     for k in range(len(accumulative_derive)):
        #         holder.append(accumulative_derive[k] * self.k[0][i])
        #     weight_change_2.append(holder)

        # weight_change_2 = np.array(weight_change_2)
        # bias_change_2 = cost_derive_1 * activation_derive_1 * activation_derive_2
        #
        # self.layers[0].update(weight_change_2 * self.learning_rate, bias_change_2 * self.learning_rate)

        for l in range(len(self.layers) - 1, 0, -1):
            layer = self.layers[l-1]
            activation_derive = layer.activation.derivative(self.k[l])
            accumulative_derive *= activation_derive

            weight_change = []
            for i in range(layer.inputs):
                holder = []
                for k in range(len(accumulative_derive)):
                    holder.append(accumulative_derive[k] * self.k[l-1][i])
                weight_change.append(holder)

            weight_change = np.array(weight_change)
            bias_change = accumulative_derive

            layer.update(weight_change * self.learning_rate, bias_change * self.learning_rate)

        return absolute_error

    def save(self):
        pass

    def load(self):
        pass


num_inputs = len(training_data[1])
layer_in = Layer(inputs=num_inputs, units=8, activation=Sigmoid, name='input')
layer2 = Layer(inputs=8, units=8, activation=Sigmoid, name='hidden1')
layer_out = Layer(inputs=8, units=1, activation=Linear, name='output')

network = Network(learning_rate=1e-3, layers=[layer_in, layer2, layer_out], cost_function=SE)
# network = Network(learning_rate=1e-3, layers=[layer1, layer4], error_function=Adaline)

for j in xrange(10000):
    # for b in xrange(100):
    # Feed forward through layers 0, 1, and 2
    error = []
    for i in range(len(training_data)):
        k2 = network.run(training_data[i])
        error.append(network.optimise(training_res[i]))
    if (j % 100) == 0:
        print "Squared Error:" + str(np.mean(np.abs(error)))


k0 = network.run(test_data)
error_rate = k0 - test_res
print(np.mean(np.abs(error_rate)))
for i, d in enumerate(test_data[-10:]):
    k0 = network.run(d)
    result_de = denormalise(k0, max_.values.tolist()[-1], min_.values.tolist()[-1])
    actual_de = denormalise(test_res[-10:][i], max_.values.tolist()[-1], min_.values.tolist()[-1])
    error = result_de - actual_de
    print('Predicted: {} | Actual: {} | Error: {}'.format(result_de, actual_de, np.abs(error)))

