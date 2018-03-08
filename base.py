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

data = np.insert(data, 0, 1., axis=1)

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

    momentum = False

    def __init__(self, inputs, units, activation=None, name=None):
        self.inputs = inputs
        self.units = units
        self.activation = activation if activation is not None else Linear
        self.name = name
        self.last_weight_change = 0
        if self.name is None:
            self.name = 'layer_{}_{}'.format(self.inputs, self.units)

        self.weights = np.random.normal(0, 0.01, (self.inputs, self.units))

    def ff(self, values):
        return self.activation.activation_fn(np.dot(values, self.weights))

    def update(self, delta):
        self.weights += delta
        if Layer.momentum:
            self.weights += self.last_weight_change * 0.9
            self.last_weight_change = delta


class Network:
    def __init__(self, learning_rate=0.1, layers=None, optimiser=None, momentum=False, annealing=None):

        if layers is None:
            raise Exception('No layers specified')

        if momentum and annealing is not None:
            raise Exception('Cannot have both momentum and annealing')

        for l in range(1, len(layers)):
            if layers[l].inputs != layers[l-1].units:
                raise Exception('{} does not have the same number of inputs ({}) as {} has units ({})'.format(
                    layers[l].name, layers[l].inputs, layers[l-1].name, layers[l-1].units
                ))

        self.momentum = momentum
        Layer.momentum = self.momentum
        self.layers = layers

        self.optimiser = optimiser

        self.optimise = self.optimiser.optimise
        self.optimiser.layers = self.layers
        self.optimiser.learning_rate = learning_rate

    def run(self, values):
        k = [np.array(values)]
        for layer in self.layers:
            k.append(layer.ff(k[-1]))
        self.optimiser.k = k
        return k[-1]

    def __str__(self):
        return 'Network: Layers: {} | LR: {} | Momentum: {}'.format(
            len(self.layers),
            self.optimiser.learning_rate,
            self.momentum
        )


num_inputs = len(training_data[1])
layer_in = Layer(inputs=num_inputs, units=8, activation=Sigmoid, name='input')
layer_1 = Layer(inputs=8, units=8, activation=Sigmoid, name='hidden')
layer_out = Layer(inputs=8, units=1, activation=Linear, name='output')
annealing = Annealing(start=1e-1, end=1e-3, epochs=5000)

network = Network(learning_rate=1e-1,
                  layers=[layer_in, layer_1, layer_out],
                  optimiser=GradientDescent(cost_function=MSE),
                  momentum=True
                  )
print(network)

for j in xrange(5000):
    error = []
    network.optimiser.learning_rate = annealing.anneal(j)
    for i in range(len(training_data)):
        _ = network.run([training_data[i]])
        error.append(network.optimise([training_res[i]]))
    if (j % 100) == 0:
        print '({}) - Error: {}'.format(j, np.mean(np.abs(error)))
        print(network.optimiser.learning_rate)

k0 = network.run(test_data)
error_rate = k0 - test_res
print(np.mean(np.abs(error_rate)))
for i, d in enumerate(test_data[-10:]):
    k0 = network.run(d)
    result_de = denormalise(k0, max_.values.tolist()[-1], min_.values.tolist()[-1])
    actual_de = denormalise(test_res[-10:][i], max_.values.tolist()[-1], min_.values.tolist()[-1])
    error = result_de - actual_de
    print('Predicted: {} | Actual: {} | Error: {}'.format(result_de, actual_de, np.abs(error)))

