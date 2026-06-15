# Importing modules
import numpy as np
import nnfs
from nnfs.datasets import spiral_data ## [nnfs stands for 'neural network from scratch'. This is a package that allows using non-linear spiral data]
nnfs.init()

# Initial Framework
class artificial_neuron:
    def __init__(self, inputs, neurons):
        self.weights = 0.01 * np.random.rand(inputs, neurons) ## [multiplying 0.01 to keep the weights small]
        self.bias = np.zeros((1, neurons)) ## [setting bias as 0 initially, so that the network will put unbiased importance to each neuron]
    def forward (self, inputs):
        self.output = np.dot(inputs, self.weights)+self.bias ## [the base formula of ANN]

# Rectified Linear Activation Function
class activation_ReLU:+
    def forward(self, inputs):
        self.output = np.maximum(0,inputs)

# Sigmoid Function
class activation_sigmoid:
    def forward(self,inputs):
        exponential = np.exp(-inputs)
        self.output = 1/(1+exponential)

# Softmax Activation Function
class activation_Softmax:
    def forward(self, inputs):
        exponential = np.exp(inputs - np.max(inputs, axis=1, keepdims=True)) ## [keepdims keeps the dimension 1]
        probabilities = exponential/np.sum(exponential, axis=1, keepdims=True)
        self.output = probabilities

# Calculating Loss with Cross-Categorical Entropy
class Loss:
    def calculate(self, output, y): # here 'y' is target values
        sample_loss = self.forward(output,y)
        data_loss = np.mean(sample_loss)
        return data_loss

class CrossCategoricalEntropy(Loss): ## [Inherited class]
    def forward(self, y_pred, y_true):
        sample = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1-1e-7) ## [ clippping the values between this range to prevent it to be 0, otherwise it will cause an error with infinity]

        if len(y_true.shape) == 1:
            confidence = y_pred_clipped[range(sample), y_true] 
        elif len(y_true.shape) == 2: ## [if the values are one-hot encoded]
            confidence = np.sum(y_pred_clipped * y_true ,axis =1)

        negative_log = -np.log(confidence)
        return negative_log
    
# Accuracy check
class Accuracy:
    def calculate(self, output, y_true):
        prediction = np.argmax(output, axis=1) ## [argmax considers the maximum value: here row-wise]
        accuracy = np.mean(prediction==y_true) ## [from the mean the likelyhood of prediction being similar to the true values can be obtained]

        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)
            
        return accuracy


X , y = spiral_data(samples=100, classes=3)

# Assigning Layers
layer1 = artificial_neuron(2,9)
activation1 = activation_ReLU()

layer2 = artificial_neuron(9,9)
activation2 = activation_sigmoid()

layer3 = artificial_neuron(9,3)
activation3 = activation_Softmax()

# Relating to each other
layer1.forward(X)
activation1.forward(layer1.output)
layer2.forward(activation1.output)
activation2.forward(layer2.output)
layer3.forward(activation2.output)
activation3.forward(layer3.output)

# Output
print(activation2.output)

# Calculating Loss
loss_function = CrossCategoricalEntropy()
loss = loss_function.calculate(activation3.output,y)
print('Loss:', loss)

# Calculatibg Accuracy
accuracy_function = Accuracy()
accuracy = accuracy_function.calculate(activation3.output,y)
print('Accuracy:', accuracy)
