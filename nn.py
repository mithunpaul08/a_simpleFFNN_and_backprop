from __future__ import division
from typing import List
import numpy as np
import math
from scipy.special import expit




class SimpleNetwork:
    """A simple feedforward network with a single hidden layer. All units in
    the network have sigmoid activation.

    """

    @classmethod
    def of(cls, n_input: int, n_hidden: int, n_output: int):
        """Creates a single-layer feedforward neural network with the given
        number of input, hidden, and output units.

        :param n_input: Number of input units
        :param n_hidden: Number of hidden units
        :param n_output: Number of output units
        :return: the neural network
        """

        def uniform(n_in, n_out):
            epsilon = math.sqrt(6) / math.sqrt(n_in + n_out)
            return np.random.uniform(-epsilon, +epsilon, size=(n_in, n_out))

        return cls(uniform(n_input, n_hidden), uniform(n_hidden, n_output))

    def __init__(self,
                 input_to_hidden_weights: np.ndarray,
                 hidden_to_output_weights: np.ndarray):
        """Creates a neural network from two weights matrices, one representing
        the weights from input units to hidden units, and the other representing
        the weights from hidden units to output units.

        :param input_to_hidden_weights: The weight matrix mapping from input
        units to hidden units
        :param hidden_to_output_weights: The weight matrix mapping from hiddden
        units to output units
        """
        self._input_to_hidden_weights=  input_to_hidden_weights
        self._hidden_to_output_weights = hidden_to_output_weights
        self.weighted_sum_layer1_after_activation = None
        self.weighted_sum_layer1_before_activation = None
        self.weighted_sum_layer2_before_activation = None
        self.weighted_sum_layer2_after_activation = None



    def predict(self, input_matrix: np.ndarray) -> np.ndarray:
        """Performs forward propagation over the neural network starting with
        the given input matriinput_matrix

        Each unit's output_before_activation should be calculated by taking a weighted sum of its
        inputs (using the appropriate weight matrix) and passing the result of
        that sum through a logistic sigmoid activation function.

        :param input_matrix: The matrix of inputs to the network, where each
        row in the matrix represents an instance for which the neural network
        should make a prediction
        :return: A matrix of predictions, where each row is the predicted
        outputs - each in the range (0, 1) - for the corresponding row in the
        input matrix.
        """
        self.weighted_sum_layer1_before_activation = np.ndarray.dot(input_matrix, self._input_to_hidden_weights)
        self.weighted_sum_layer1_after_activation = expit(self.weighted_sum_layer1_before_activation)
        self.weighted_sum_layer2_before_activation = np.ndarray.dot(self.weighted_sum_layer1_after_activation, self._hidden_to_output_weights)
        self.weighted_sum_layer2_after_activation = expit(self.weighted_sum_layer2_before_activation)
        return self.weighted_sum_layer2_after_activation


    def predict_zero_one(self, input_matrix: np.ndarray) -> np.ndarray:
        """Performs forward propagation over the neural network starting with
        the given input matrix, and converts the outputs to binary (0 or 1).

        Outputs will be converted to 0 if they are less than 0.5, and converted
        to 1 otherwise.

        :param input_matrix: The matrix of inputs to the network, where each
        row in the matrix represents an instance for which the neural network
        should make a prediction
        :return: A matrix of predictions, where each row is the predicted
        outputs - each either 0 or 1 - for the corresponding row in the input
        matrix.
        """
        prediction=self.predict(input_matrix)
        prediction_zero_one_format=np.where(prediction<0.5,0,1)
        return prediction_zero_one_format



    def gradients(self,
                  input_matrix: np.ndarray,
                  output_matrix: np.ndarray) -> List[np.ndarray]:
        """Performs back-propagation to calculate the gradients for each of
        the weight matrices.

        This method first performs a pass of forward propagation through the
        network, keeping track of the weighted sums before the activation
        function (at layer l, we call such a vector a_l) and the values
        after the activation function (at layer l, we call such a vector h_l,
        and similarly refer to the input as h_0).

        Then for each input example, the method applies the following
        calculations, where × is matrix multiplication, ⊙ is element-wise
        product, and ⊤ is matrix transpose:

        1. g_2 = ((h_2 - y) ⊙ sigmoid'(a_2))⊤
        2. hidden-to-output weights gradient += (g_2 × h_1)⊤
        3. g_1 = (((hidden-to-output weights) × g_2)⊤ ⊙ sigmoid'(a_1))⊤
        4. input-to-hidden weights gradient += (g_1 × h_0)⊤

        When all input examples have applied their updates to the gradients,
        each entire gradient should be divided by the number of input examples.

        :param input_matrix: The matrix of inputs to the network, where each
        row in the matrix represents an instance for which the neural network
        should make a prediction
        :param output_matrix: A matrix of expected outputs, where each row is
        the expected outputs - each either 0 or 1 - for the corresponding row in
        the input matrix.
        :return: two matrices of gradients, one for the input-to-hidden weights
        and one for the hidden-to-output weights
        """

        # 1. g_2 = ((h_2 - y) ⊙ sigmoid'(a_2))⊤
        prediction = self.predict(input_matrix)
        diff=prediction - output_matrix
        reverse_acitvation2=self.sigmoid_gradient(self.weighted_sum_layer2_before_activation)
        g_2 = (np.multiply(diff,reverse_acitvation2)).T

        #2. hidden-to-output weights gradient += (g_2 × h_1)⊤
        hidden_to_output_weights_gradient= np.ndarray.dot(g_2,self.weighted_sum_layer1_after_activation).T

        # 3. g_1 = (((hidden-to-output weights) × g_2)⊤ ⊙ sigmoid'(a_1))⊤
        reverse_acitvation1 = self.sigmoid_gradient(self.weighted_sum_layer1_before_activation)
        part1=np.ndarray.dot(self._hidden_to_output_weights, g_2)
        g_1 = np.multiply((part1.T),reverse_acitvation1).T

        #4. input-to-hidden weights gradient += (g_1 × h_0)⊤
        input_to_hidden_weights_gradient= (np.ndarray.dot(g_1,input_matrix).T)

        no_of_input_examples=input_matrix.shape[0]
        return [input_to_hidden_weights_gradient/no_of_input_examples,hidden_to_output_weights_gradient/no_of_input_examples]

    def train(self,
              input_matrix: np.ndarray,
              output_matrix: np.ndarray,
              iterations: int = 10,
              learning_rate: float = 0.1) -> None:
        """Trains the neural network on an input matrix and an expected output
        matrix.

        Training should repeatedly (`iterations` times) calculate the gradients,
        and update the model by subtracting the learning rate times the
        gradients from the model weight matrices.

        :param input_matrix: The matrix of inputs to the network, where each
        row in the matrix represents an instance for which the neural network
        should make a prediction
        :param output_matrix: A matrix of expected outputs, where each row is
        the expected outputs - each either 0 or 1 - for the corresponding row in
        the input matrix.
        :param iterations: The number of gradient descent steps to take.
        :param learning_rate: The size of gradient descent steps to take, a
        number that the gradients should be multiplied by before updating the
        model weights.

        """
        for epoch in range(0, iterations):
                input_to_hidden_gradient,hidden_to_output_gradient = self.gradients(input_matrix, output_matrix)
                self._input_to_hidden_weights=self._input_to_hidden_weights- (learning_rate*input_to_hidden_gradient)
                self._hidden_to_output_weights = self._hidden_to_output_weights - (learning_rate * hidden_to_output_gradient)

    def sigmoid(self,x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_gradient(self,x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))

