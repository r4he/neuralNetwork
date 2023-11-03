import numpy as np

class neuralNetwork:
    def __init__(self, learning_rate):
        # randomly generate weights & bias
        self.weights = np.array([np.random.randn(), np.random.randn()])
        self.bias = np.random.randn()
        self.learning_rate = learning_rate

    # limits outputs between 0-1
    def _sigmoid(self, x):
        return 1/(1+np.exp(-x))

    def _sigmoid_deriv(self, x):
        return self._sigmoid(x)*(1-self._sigmoid(x))

    def predict(self, input_vector):
        layer_1 = np.dot(input_vector, self.weights) + self.bias
        layer_2 = self._sigmoid(layer_1)
        return layer_2

    def _compute_gradients(self, input_vector, target):
        layer_1 = np.dot(input_vector, self.weights) + self.bias
        layer_2 = self._sigmoid(layer_1)
        prediction = layer_2
        # note: a,b are constants
        # derivative of error in terms of prediction [(x-a)^2]' = (x-a)*2*1
        derror_dprediction = 2 * (prediction - target)

        # derivative of prediction in terms of layer 1 [sigmoid(x)]' = sigmoid(x)*(1-sigmoid(x))
        dprediction_dlayer1 = self._sigmoid_deriv(prediction)

        # derivative of layer 1 in terms of bias [dot(a,b)+x]'=1
        dlayer1_dbias = 1
        # derivative of layer 1 in terms of weights [dot(a,x)+b]'=dot([a]',x)+dot(a,[x]')
        dlayer1_dweights = (input_vector*1)+(0*self.weights)

        # total bias error derivative
        derror_dbias = (derror_dprediction*dprediction_dlayer1*dlayer1_dbias)
        # total weights error derivative
        derror_dweights = (derror_dprediction*dprediction_dlayer1*dlayer1_dweights)

        return derror_dbias, derror_dweights

    # update weights & bias using errors
    def _update_parameters(self,derror_dbias,derror_dweights):
        self.weights = self.weights - (self.learning_rate*derror_dweights)
        self.bias = self.bias - (self.learning_rate*derror_dbias)