import numpy as np
X = [[2.5, 5.6, 0.5],
     [1.4, 2.5, 1.1],
     [2.3, 4.7, 3.3]]

# Initial Framework
class artificial_neuron:
    def __init__(self, weights, neurons):
        self.weights = np.random.rand(weights, neurons)
        self.bias = np.random.rand(neurons)
    def forward (self, X):
        self.output = np.dot(X, self.weights)+self.bias

# Rectified Linear Activation Function
class activation_ReLU:
    def forward (self, X):
        self.output = np.maximum(0,X)

# Assigning Layers
layer1 = artificial_neuron(3,7)
layer2 = artificial_neuron(7,2)
activation = activation_ReLU()

# Relating to each other
layer1.forward(X)
activation.forward(layer1.output)
layer2.forward(activation.output)

# Output
print(layer2.output)
