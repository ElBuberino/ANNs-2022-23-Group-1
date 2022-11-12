import numpy as np
import matplotlib.pyplot as plt



def relu(x):
    return np.maximum(0, x)

def reluDerivative(x):
    return x > 0

def mean_squared_error(target, output):
    return (1/2) * (np.square(target - output))

def mseDerivative(target, output):
    return output - target

def transpose_Vector(inp):
    return np.transpose(inp)

class Layer:
    def __init__(self, input_units, n_units):
        self.input_units = input_units
        self.n_units = n_units
        self.weights = np.random.rand(n_units, input_units)
        self.bias = np.zeros((n_units, 1))
        self.preactivation = 0
        self.target = None
        self.output = 0
        self.inputs = None

    def forward_step(self, inputs):
        self.preactivation = self.weights @ inputs + self.bias
        return relu(self.preactivation)

    def compute_weight_gradient(self,output):
        gradient_weights = (reluDerivative(self.preactivation) * mseDerivative(self.target, output)) @ np.transpose(self.inputs)
        return gradient_weights

    def compute_input_gradient(self,output):
        gradient_input = np.matrix.transpose(self.weights) @ ((reluDerivative(self.preactivation)) * mseDerivative(self.target, output))
        return gradient_input

    def compute_bias_gradient(self,output):
        gradient_bias = reluDerivative(self.preactivation) * mseDerivative(self.target, output)
        return gradient_bias

    def backward_step(self,inputs, target):
        self.inputs = inputs
        self.target = target
        output = self.forward_step(self.inputs)
        update_weights = self.compute_weight_gradient(output)
        update_bias = self.compute_bias_gradient(output)
        self.weights -= 0.01 * update_weights
        self.bias -= 0.01 * update_bias
        return self.compute_input_gradient(output)


class MLP:
    def __init__(self):
        self.layer_1 = Layer(1, 10)
        self.layer_2 = Layer(10, 1)
        self.layer_1_activation = None
        self.layer_2_activation = None
        self.update_layer_1 = None
        self.update_layer_2 = None
        self.output = None

    def forwardprop_step(self, inputs):
        self.layer_1_activation = self.layer_1.forward_step(inputs)
        self.layer_2_activation = self.layer_2.forward_step(self.layer_1_activation)
        self.output = self.layer_2_activation
        return self.output

    def backprop_step(self, inputs, target):
        self.update_layer_2 = self.layer_2.backward_step(self.layer_1_activation, target)
        self.update_layer_1 = self.update_layer_2 * self.layer_1.backward_step(inputs, target)
        return self.update_layer_2, self.update_layer_1

ann1 = MLP()
print(ann1.forwardprop_step([[5]]))
print(ann1.backprop_step([[5]], [[1]]))


""" Training and visualization is still now working"""
x = np.random.uniform(0,1,100)

t = [i**3-i**2 for i in x]

epochs = []
losses = []

for epoch in range(1000):
    epochs.append(epoch)

    loss_counter = 0

    for i in range(10):
        x_num = x[i]
        t_num = t[i]

        ann1.forwardprop_step([x_num])
        ann1.backprop_step([x_num], [t_num])

        loss_counter += (t_num - ann1.output)**2

    losses.append(loss_counter)


plt.figure()
plt.plot(epochs,losses)
plt.xlabel("Training Steps")
plt.ylabel("Loss")
plt.show()