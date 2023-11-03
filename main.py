import numpy as np
learning_rate = 0.1
from neural_class import neuralNetwork

neural_network = neuralNetwork(learning_rate)
input_vector = np.array([1.66, 1.56])
target = np.array([1.0])
# initial prediction
prediction = neural_network.predict(input_vector)
print(prediction)

# update parameters until prediction is above 0.5
while(prediction<0.5):
    neural_network._update_parameters(*neural_network._compute_gradients(input_vector, target))
    prediction = neural_network.predict(input_vector)
    print(prediction)

# new weights & bias
print(neural_network.weights, " ", neural_network.bias)
