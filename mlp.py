"""
Does depth hurt trainability? A from-scratch MLP in pure Python.

Trains narrow MLPs (fixed width, varying depth) on a small regression problem
and records how many epochs each takes to reach a target training loss, plus
per-layer gradient sizes.

Usage:
    python mlp.py --quick                    # sanity check: depth 1 and 12, one seed
    python mlp.py                            # full sigmoid experiment
    python mlp.py --activation relu          # ReLU variant (He init, linear output)
    python mlp.py --activation leaky_relu
"""
import argparse
import json
import math
import os
import random
import time

LEAKY_SLOPE = 0.01


# ---------------- model ----------------

class Neuron:
    def __init__(self, num_inputs, activation, init):
        self.activation = activation
        if init == "uniform":                       # used with sigmoid
            self.weights = [random.uniform(-1, 1) for _ in range(num_inputs)]
            self.bias = random.uniform(-1, 1)
        elif init == "he":                          # used with ReLU variants
            scale = math.sqrt(2 / num_inputs)
            self.weights = [random.gauss(0, scale) for _ in range(num_inputs)]
            self.bias = 0.0
        else:
            raise ValueError(f"unknown init: {init}")

    def _activate(self, z):
        if self.activation == "sigmoid":
            if z >= 0:
                return 1 / (1 + math.exp(-z))
            e = math.exp(z)                         # numerically stable for negative z
            return e / (1 + e)
        if self.activation == "relu":
            return max(0.0, z)
        if self.activation == "leaky_relu":
            return z if z > 0 else LEAKY_SLOPE * z
        if self.activation == "linear":
            return z
        raise ValueError(f"unknown activation: {self.activation}")

    def derivative(self):
        """Derivative of the activation at the last forward pass."""
        if self.activation == "sigmoid":
            return self.output * (1 - self.output)
        if self.activation == "relu":
            return 1.0 if self.z > 0 else 0.0
        if self.activation == "leaky_relu":
            return 1.0 if self.z > 0 else LEAKY_SLOPE
        return 1.0                                  # linear

    def forward(self, inputs):
        self.inputs = inputs
        self.z = sum(w * i for w, i in zip(self.weights, inputs)) + self.bias
        self.output = self._activate(self.z)
        return self.output

    def backward(self, delta):
        self.weight_gradients = [delta * i for i in self.inputs]
        self.bias_gradient = delta
        return [delta * w for w in self.weights]    # gradient w.r.t. this neuron's inputs

    def update(self, learning_rate):
        for i in range(len(self.weights)):
            self.weights[i] -= learning_rate * self.weight_gradients[i]
        self.bias -= learning_rate * self.bias_gradient


class Layer:
    def __init__(self, num_inputs, num_neurons, activation, init):
        self.neurons = [Neuron(num_inputs, activation, init) for _ in range(num_neurons)]

    def forward(self, inputs):
        return [neuron.forward(inputs) for neuron in self.neurons]

    def backward(self, deltas):
        input_grads = [0.0] * len(self.neurons[0].weights)
        for neuron, delta in zip(self.neurons, deltas):
            for i, grad in enumerate(neuron.backward(delta)):
                input_grads[i] += grad
        return input_grads

    def update(self, learning_rate):
        for neuron in self.neurons:
            neuron.update(learning_rate)


class MLP:
    def __init__(self, num_inputs, layer_sizes,
                 hidden_activation="sigmoid", output_activation="sigmoid"):
        init = "uniform" if hidden_activation == "sigmoid" else "he"
        self.layers = []
        input_size = num_inputs
        last = len(layer_sizes) - 1
        for i, size in enumerate(layer_sizes):
            activation = output_activation if i == last else hidden_activation
            self.layers.append(Layer(input_size, size, activation, init))
            input_size = size

    def forward(self, inputs):
        for layer in self.layers:
            inputs = layer.forward(inputs)
        return inputs[0] if len(inputs) == 1 else inputs

    def backward(self, targets):
        """Backpropagate squared-error loss for one sample (call forward first)."""
        output_layer = self.layers[-1]
        n = len(output_layer.neurons)
        deltas = [
            (2 / n) * (neuron.output - target) * neuron.derivative()
            for neuron, target in zip(output_layer.neurons, targets)
        ]
        for i in range(len(self.layers) - 1, -1, -1):
            input_grads = self.layers[i].backward(deltas)
            if i > 0:
                deltas = [
                    grad * neuron.derivative()
                    for grad, neuron in zip(input_grads, self.layers[i - 1].neurons)
                ]

    def update(self, learning_rate):
        for layer in self.layers:
            layer.update(learning_rate)

    def count_params(self):
        return sum(len(n.weights) + 1 for l in self.layers for n in l.neurons)

    def grad_norms(self):
        """L2 norm of the current gradients, one number per layer."""
        return [
            math.sqrt(
                sum(g ** 2 for n in l.neurons for g in n.weight_gradients)
                + sum(n.bias_gradient ** 2 for n in l.neurons)
            )
            for l in self.layers
        ]

    def train(self, X, targets, learning_rate, epochs,
              print_every=0, log_every=100, stop_loss=None):
        """Per-sample SGD, fixed sample order. Returns (loss history, gradient history, stop epoch)."""
        history = []
        grad_history = []
        stop_epoch = None                           # stays None if stop_loss is never reached

        for epoch in range(1, epochs + 1):
            log_now = (epoch % log_every == 0)
            if log_now:
                totals = [0.0] * len(self.layers)

            for x, target in zip(X, targets):
                self.forward(x)
                self.backward([target])
                if log_now:
                    totals = [t + g for t, g in zip(totals, self.grad_norms())]
                self.update(learning_rate)

            if log_now:
                grad_history.append([t / len(X) for t in totals])

            preds = [self.forward(x) for x in X]
            mean_loss = sum((p - t) ** 2 for p, t in zip(preds, targets)) / len(X)
            history.append(mean_loss)

            if print_every and (epoch == 1 or epoch % print_every == 0):
                print(f"Epoch {epoch:5d} | loss {mean_loss:.6f}")

            if stop_loss is not None and mean_loss < stop_loss:
                stop_epoch = epoch
                break

        return history, grad_history, stop_epoch


# ---------------- data ----------------

def target_function(x):
    return (x[0] * x[1] + x[0] ** 3 + x[1] ** 3) / 3


X_TRAIN = [
    [0.1, 0.2], [0.2, 0.8], [0.3, 0.5], [0.4, 0.9], [0.5, 0.1],
    [0.6, 0.7], [0.7, 0.3], [0.8, 0.6], [0.9, 0.4], [0.9, 0.9],
]
Y_TRAIN = [target_function(x) for x in X_TRAIN]

_test_rng = random.Random(123)                      # separate generator: same test set for every run
X_TEST = [[_test_rng.random() for _ in range(2)] for _ in range(200)]
Y_TEST = [target_function(x) for x in X_TEST]

_train_mean = sum(Y_TRAIN) / len(Y_TRAIN)
TRAIN_BASELINE = sum((t - _train_mean) ** 2 for t in Y_TRAIN) / len(Y_TRAIN)  # best constant predictor, train
TEST_BASELINE = sum((t - _train_mean) ** 2 for t in Y_TEST) / len(Y_TEST)     # same constant, test


def mse(model, X, y):
    preds = [model.forward(x) for x in X]
    return sum((p - t) ** 2 for p, t in zip(preds, y)) / len(X)


# ---------------- experiment ----------------

ACTIVATION_DEFAULTS = {
    #               output layer   learning rate
    "sigmoid":    {"output": "sigmoid", "lr": 1.0},
    "relu":       {"output": "linear",  "lr": 0.05},   # lr not tuned
    "leaky_relu": {"output": "linear",  "lr": 0.05},   # lr not tuned
}


def run_experiment(seed, hidden_layers, activation, lr, epochs, stop_loss):
    random.seed(seed)                               # seed BEFORE building the net
    output_activation = ACTIVATION_DEFAULTS[activation]["output"]
    model = MLP(2, hidden_layers + [1], activation, output_activation)
    history, grad_history, stop_epoch = model.train(
        X_TRAIN, Y_TRAIN, lr, epochs, stop_loss=stop_loss)
    return {
        "seed": seed,
        "depth": len(hidden_layers),
        "width": hidden_layers[0],
        "activation": activation,
        "lr": lr,
        "max_epochs": epochs,
        "stop_loss": stop_loss,
        "params": model.count_params(),
        "stop_epoch": stop_epoch,                   # None = never reached stop_loss
        "train_loss": history[-1],
        "test_loss": mse(model, X_TEST, Y_TEST),
        "train_baseline": TRAIN_BASELINE,
        "baseline": TEST_BASELINE,
        "grad_history": grad_history,               # one row per 100 epochs, one number per layer
        "history": history[::10],                   # every 10th epoch, keeps files small
    }


def main():
    parser = argparse.ArgumentParser(description="Depth vs trainability experiment.")
    parser.add_argument("--activation", choices=list(ACTIVATION_DEFAULTS), default="sigmoid")
    parser.add_argument("--depths", type=int, nargs="+", default=[1, 2, 4, 8, 12])
    parser.add_argument("--width", type=int, default=4)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=50000)
    parser.add_argument("--lr", type=float, default=None, help="default depends on activation")
    parser.add_argument("--stop-loss", type=float, default=0.001)
    parser.add_argument("--out-dir", default="results")
    parser.add_argument("--quick", action="store_true",
                        help="depth 1 and 12, one seed, nothing saved")
    args = parser.parse_args()

    lr = args.lr if args.lr is not None else ACTIVATION_DEFAULTS[args.activation]["lr"]

    if args.quick:
        for depth in [1, 12]:
            t = time.time()
            r = run_experiment(0, [args.width] * depth, args.activation, lr,
                               args.epochs, args.stop_loss)
            print(f"depth {depth:2d} | params {r['params']} | stopped at {r['stop_epoch']} | "
                  f"train {r['train_loss']:.5f} | test {r['test_loss']:.5f} | "
                  f"baseline {r['baseline']:.5f} | {time.time() - t:.1f} s")
        return

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, f"{args.activation}_lr{lr}.json")
    results = []
    for depth in args.depths:
        for seed in range(args.seeds):
            r = run_experiment(seed, [args.width] * depth, args.activation, lr,
                               args.epochs, args.stop_loss)
            results.append(r)
            print(f"depth {depth:2d} seed {seed} | params {r['params']} | "
                  f"stopped at {r['stop_epoch']} | test {r['test_loss']:.5f} "
                  f"(baseline {r['baseline']:.5f})")
            with open(out_path, "w") as fh:         # save after every run in case of a crash
                json.dump(results, fh)
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()