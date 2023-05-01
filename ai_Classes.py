import numpy as np
import matplotlib.pyplot as plt

class dense:
    def __init__(self, input, neuron):
        self.weights = 0.01 * np.random.randn(input, neuron)
        self.biases = np.zeros([1, neuron])
        
    def forward(self, inputs):
        self.inputs = inputs
        self.outputs = np.dot(inputs, self.weights) + self.biases
        
    def backward(self, dval):
        self.dweights = np.dot(self.inputs.T, dval)
        self.dbiases = np.sum(dval, axis = 0, keepdims = True)
    
class costMAE:
    def forward(self, prediction, real):
        return np.mean(np.abs(prediction - real))
    
    def backward(self, prediction, real):
        self.dinputs = np.sign(prediction - real) / prediction.size
    
class costMSE:
    def forward(self, prediction, real):
        return np.mean(np.square(prediction - real))
    
    def backward(self, prediction, real):
        self.dinputs = (2 / prediction.size) * (prediction - real)

class SGD_Optimizer:
    def __init__(self, learning_rate, mu = 0):
        self.learnRate = learning_rate
        self.mu = mu
    
    def update_params(self, layer):
        if hasattr(layer, "v_weights") == False:
            layer.v_weights = np.zeros_like(layer.weights)
            layer.v_bias = np.zeros_like(layer.biases)
            
        layer.v_weights = (self.mu * layer.v_weights) + (self.learnRate * -layer.dweights)
        layer.v_bias = (self.mu * layer.v_bias) + (self.learnRate * -layer.dbiases)
        layer.weights += layer.v_weights
        layer.biases += layer.v_bias

class Learning_Rate_Decayer:
    def __init__(self, optimizer, decay_factor):
        self.epochs = 0
        self.optimizer = optimizer
        self.initial_lr = optimizer.lr
        self.decay_factor = decay_factor
    
    def update_learning_rate(self):
        self.epochs += 1
        self.optimizer.lr = self.initial_lr / (1 + (self.decay_factor * self.epochs))

class Adam_Optimizer:
    def __init__(self, learning_rate, beta1=0.9, beta2=0.999, eps=1e-8):    # rho is decay rate
        self.epochs = 0
        self.lr = learning_rate
        self.beta1 = beta1    # beta1 is coefficient of friction for momentum
        self.beta2 = beta2    # beta2 is decay rate for RMSProp
        self.eps = eps     # epsilon
    
    def update_params(self, layer):    # dense layer
        self.epochs += 1
        # if layer does not have the attribute "v_weights", the layer also does not have
        # the attributes "v_biases", "cache_weights", and "cache_biases"
        # we will give the let's initialize those attributes with cache as 0
        if hasattr(layer, "v_weights") == False:
            layer.v_weights = np.zeros_like(layer.weights)
            layer.v_biases = np.zeros_like(layer.biases)
            layer.cache_weights = np.zeros_like(layer.weights)
            layer.cache_biases = np.zeros_like(layer.biases)
        
        # velocities
        layer.v_weights = (layer.v_weights * self.beta1) + ((1 - self.beta1) * layer.dweights * 2)
        layer.v_biases = (layer.v_biases * self.beta1) + ((1 - self.beta1) * layer.dbiases * 2)

        # velocity corrections
        layer.v_weights_corrected = layer.v_weights / (1 - (self.beta1 ** self.epochs))
        layer.v_biases_corrected = layer.v_biases / (1 - (self.beta1 ** self.epochs))

        # caches
        layer.cache_weights = (layer.cache_weights * self.beta2) + ((1 - self.beta2) * layer.dweights ** 2)
        layer.cache_biases = (layer.cache_biases * self.beta2) + ((1 - self.beta2) * layer.dbiases ** 2)

        # cache corrections
        layer.cache_weights_corrected = layer.cache_weights / (1 - (self.beta2 ** self.epochs))
        layer.cache_biases_corrected = layer.cache_biases / (1 - (self.beta2 ** self.epochs))

        # update
        layer.weights += (self.lr / (np.sqrt(layer.cache_weights_corrected) + self.eps)) * -layer.v_weights_corrected
        layer.biases += (self.lr / (np.sqrt(layer.cache_biases_corrected) + self.eps)) * -layer.v_biases_corrected

class RMSProp_Optimizer:
    def __init__(self, learning_rate, rho = 0.9, eps = 1e-7):
        self.lr = learning_rate
        self.rho = rho
        self.eps = eps
        
    def update_params(self, layer):
        if hasattr(layer, "cache_weights") == False:
            layer.cache_weights = np.zeros_like(layer.weights)
            layer.cache_biases = np.zeros_like(layer.biases)
        layer.cache_weights = (layer.cache_weights * self.rho) + ((1 - self.rho) * layer.dweights ** 2)
        layer.cache_biases = (layer.cache_biases * self.rho) + ((1 - self.rho) * layer.dbiases ** 2)
        
        layer.weights += (self.lr / (np.sqrt(layer.cache_weights) + self.eps)) * -layer.dweights
        layer.biases += (self.lr / (np.sqrt(layer.cache_biases) + self.eps)) * -layer.dbiases