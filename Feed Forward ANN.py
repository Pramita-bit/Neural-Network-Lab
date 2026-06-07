# Importing modules
import numpy as np
import nnfs
from nnfs.datasets import spiral_data
nnfs.init()

# Initial Framework
class artificial_neuron:
    def __init__(self, inputs, neurons):
        self.weights = 0.01 * np.random.rand(inputs, neurons)
        self.bias = np.random.rand(1, neurons)
    def forward (self, inputs):
        self.output = np.dot(inputs, self.weights)+self.bias

# Rectified Linear Activation Function
class activation_ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0,inputs)

# Softmax Activation Function
class activation_Softmax:
    def forward(self, inputs):
        exponential = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exponential/np.sum(exponential, axis=1, keepdims=True)
        self.output = probabilities

X , y = spiral_data(samples=100, classes=3)

# Assigning Layers
layer1 = artificial_neuron(2,9)
activation1 = activation_ReLU()

layer2 = artificial_neuron(9,3)
activation2 = activation_Softmax()

# Relating to each other
layer1.forward(X)
activation1.forward(layer1.output)
layer2.forward(activation1.output)
activation2.forward(layer2.output)

# Output
print(activation2.output)
