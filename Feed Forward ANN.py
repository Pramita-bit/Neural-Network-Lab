# Importing modules
import numpy as np
import nnfs
from nnfs.datasets import spiral_data
nnfs.init()

# Initial Framework
class artificial_neuron:
    def __init__(self, inputs, neurons):
        self.weights = 0.01 * np.random.rand(inputs, neurons)
        self.bias = np.zeros((1, neurons))
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

# Calculating Loss with Cross-Categorical Entropy
class Loss:
    def calculate(self, output, y): # here 'y' is target values
        sample_loss = self.forward(output,y)
        data_loss = np.mean(sample_loss)
        return data_loss

class CrossCategoricalEntropy(Loss):
    def forward(self, y_pred, y_true):
        sample = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1-1e-7)

        if len(y_true.shape) == 1:
            confidence = y_pred_clipped[range(sample), y_true] 
        elif len(y_true.shape) == 2:
            confidence = np.sum(y_pred_clipped * y_true ,axis =1)

        negative_log = -np.log(confidence)
        return negative_log
    
# Accuracy check
class Accuracy:
    def calculate(self, output, y_true):
        prediction = np.argmax(output, axis=1)
        accuracy = np.mean(prediction==y_true)

        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)
            
        return accuracy


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

# Calculating Loss
loss_function = CrossCategoricalEntropy()
loss = loss_function.calculate(activation2.output,y)
print('Loss:', loss)

accuracy_function = Accuracy()
accuracy = accuracy_function.calculate(activation2.output,y)
print('Accuracy:', accuracy)
