from data_processing import get_all_data
from data_processing import denormalise
from activation import *
from error import *


class NoDataException(Exception):
    pass


class NotLayerException(Exception):
    pass


def get_data_set(training_percentage=.6, validation_percentage=0., add_biases=True):
    data, max_, min_ = get_all_data()
    np.random.seed(1)
    np.random.shuffle(data)

    if add_biases:
        data = np.insert(data, 0, 1., axis=1)

    data_length = len(data)
    training_size = int(data_length * training_percentage)
    validation_size = int(data_length * validation_percentage)

    training = data[:training_size]
    if validation_percentage > 0:
        validation = data[training_size:training_size + validation_size]
        test = data[training_size + validation_size:]
    else:
        validation = []
        test = data[training_size:]

    training_data = training[:, :-1]
    training_res = training[:, -1:]
    test_data = test[:, :-1]
    test_res = test[:, -1:]
    validation_data = validation[:, :-1]
    validation_res = validation[:, -1:]

    return training_data, training_res, test_data, test_res, validation_data, validation_res, (max_, min_)


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

    def feed_forward(self, values):
        return self.activation.activation_fn(np.dot(values, self.weights))

    def update(self, delta):
        self.weights += delta
        if Layer.momentum:
            self.weights += self.last_weight_change * 0.9
            self.last_weight_change = delta

    def __str__(self):
        return 'Layer {} | Inputs: {} | Units: {} | Activation: {}'.format(
            self.name,
            self.inputs,
            self.units,
            self.activation
        )


class Network:
    def __init__(self, learning_rate=0.1, layers=None, optimiser=None, momentum=False):

        if layers is None:
            raise Exception('No layers specified')

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
            k.append(layer.feed_forward(k[-1]))
        self.optimiser.k = k
        return k[-1]

    def __str__(self):
        return 'Network layers: {} | Layers: {} | LR: {} | Momentum: {}'.format(
            len(self.layers) + 1,
            [str(layer) for layer in self.layers],
            self.optimiser.learning_rate,
            self.momentum
        )


training_set, training_set_res, test_set, test_set_res, validation_set, validation_set_res, max_min = get_data_set(validation_percentage=.2)

num_inputs = len(training_set[1])
layer_in = Layer(inputs=num_inputs, units=12, activation=LeakyReLu, name='input')
layer_out = Layer(inputs=12, units=1, activation=Linear, name='output')
annealing = Annealing(start=1e-1, end=1e-4, epochs=10000)

network = Network(learning_rate=1e-1,
                  layers=[layer_in, layer_out],
                  optimiser=GradientDescent(cost_function=MSE),
                  momentum=True
                  )
print(network)

# for j in xrange(10000):
#     error = []
#     network.optimiser.learning_rate = annealing.anneal(j)
#     for i in range(len(training_set)):
#         _ = network.run([training_set[i]])
#         error.append(network.optimise([training_set_res[i]]))
#     if (j % 100) == 0:
#         print '({}) - Error: {}'.format(j, np.mean(np.abs(error)))
#         print(network.optimiser.learning_rate)

epoch = 0
previous_validation_error = 0
while True and epoch < 20000:
    error = []
    validation_error = []
    network.optimiser.learning_rate = annealing.anneal(epoch)
    for i in range(len(training_set)):
        _ = network.run([training_set[i]])
        error.append(network.optimise([training_set_res[i]]))
    if epoch == 500:
        previous_validation_error = np.mean(np.abs(error))
    if (epoch - 500) % 200 == 0 and epoch > 500:
        print('Testing the validation set')
        validation_result = network.run(validation_set)
        validation_error.append(validation_result - validation_set_res)
        if np.mean(np.abs(error)) > previous_validation_error:
            print('Validation set gone up')
            break

    if epoch % 100 == 0:
        print '({}) - Error: {}'.format(epoch, np.mean(np.abs(error)))
        print(network.optimiser.learning_rate)
    epoch += 1

k1 = network.run(training_set)
error_rate_1 = k1 - training_set_res
print(np.mean(np.abs(error_rate_1)))
k0 = network.run(test_set)
error_rate = k0 - test_set_res
print(np.mean(np.abs(error_rate)))
for i, d in enumerate(test_set[-10:]):
    k0 = network.run(d)
    result_de = denormalise(k0, max_min[0].values.tolist()[-1], max_min[1].values.tolist()[-1])
    actual_de = denormalise(test_set_res[-10:][i], max_min[0].values.tolist()[-1], max_min[1].values.tolist()[-1])
    error = result_de - actual_de
    print('Predicted: {} | Actual: {} | Error: {}'.format(result_de, actual_de, np.abs(error)))

