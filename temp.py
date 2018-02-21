import numpy as np
from activation import sigmoid

data = [(1, 4, 1), (2, 9, 1), (5, 6, 1), (4, 5, 1), (6, 0.7, 1), (1, 1.5, 1)]
res = [1, 1, 1, 1, 0, 0]

w0 = np.random.random((3, 2))
w1 = np.random.random((2, 1))


def feed_forward(inputs, weights):
    out = []
    for w in range(len(weights[1])):
        temp = 0
        for i, ipt in enumerate(inputs):
            temp += ipt * weights[i][w]
        out.append(sigmoid(temp))
    return out


def calc_delta_output(result, correct):
    inv = inverse(result)
    return (correct - result) * inv


def calc_delta(result, weight, output_delta):
    inv = inverse(result)
    return (weight * output_delta) * inv


def inverse(x):
    return x * (1 - x)


def get_deltas(result, correct, w1):
    output_delta = calc_delta_output(result, correct)
    deltas = []
    for weight in w1:
        deltas.append(calc_delta(result, weight, output_delta)[0])

    return output_delta, deltas


def update_weights(inputs, w0, w1, output_delta, deltas, ans):

    # update the later weights first
    for w in range(len(w1[1])):
        for i, ipt in enumerate(ans):
            old_weight = w1[i][w]
            w1[i][w] = old_weight + (0.1 * output_delta) * ipt

    for w in range(len(w0[1])):
        for i, ipt in enumerate(inputs):
            old_weight = w0[i][w]
            w0[i][w] = old_weight + (0.1 * deltas[w]) * ipt


for _ in range(5000):
    for j, i in enumerate(data):
        ans = feed_forward(i, w0)
        ans_out = feed_forward(ans, w1)[0]
        output_delta, deltas = get_deltas(ans_out, res[j], w1)
        update_weights(i, w0, w1, output_delta, deltas, ans)

count = 0
for jj, test in enumerate(data):
    a1 = feed_forward(test, w0)
    a2 = feed_forward(a1, w1)[0]
    temp_ = 1 if a2 > 0.5 else -1
    print(a2)
    if temp_ == res[jj]:
        count += 1

print(count)
print(w0)
print(w1)
