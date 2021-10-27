########################################################################################################################
# A neural network.
#
# Code adaption of [1].
#
# [1] https://github.com/rb-rl/DDPG/network.py.
#
########################################################################################################################

import torch

from torch import Tensor
from torch.nn import BatchNorm1d, Linear, Module
from torch.nn.functional import relu, tanh
from typing import Tuple

# TOPOLOGY -------------------------------------------------------------------------------------------------------------

# The activation function used in all but the last layer (for both actor and critic).
ACTIVATION_FUNCTION = "relu"

# The number of hidden layers of the actor network.
ACTOR_NUMBER_HIDDEN_LAYERS = 2

# The number of hidden neurons per layer of the actor network.
ACTOR_NUMBER_HIDDEN_NEURONS_PER_LAYER = 64

# The number of hidden layers of the critic network.
CRITIC_NUMBER_HIDDEN_LAYERS = 3

# The number of hidden neurons per layer of the critic network.
CRITIC_NUMBER_HIDDEN_NEURONS_PER_LAYER = 64

# Use batch normalization on the sensor inputs?
USE_BATCH_NORMALIZATION = False

# DEVELOPMENT ----------------------------------------------------------------------------------------------------------

# Is the fast development mode activated which reloads imports?
#
# The advantage of the fast development mode is that one does not have to restart Python from scratch fear each
# development increment which makes the development faster.
FAST_DEVELOPMENT_MODE = True


class NeuralNetwork(Module):
    """
    A neural network based on fully connected layers.

    Note that the inputs are concatenated when fed into the neural network.
    """

    def __init__(self, numbers_inputs: Tuple[int, ...], number_outputs: int,
                 number_hidden_layers: int, number_hidden_neurons_per_layer: int,
                 number_normalized_inputs: int, normalized_output: bool):
        """
        Initialize the fully connected layers.

        Args:
            numbers_inputs: The numbers of input neurons.
            number_outputs: The number of output neurons.
            number_hidden_layers: The number of hidden layers.
            number_hidden_neurons_per_layer: The number of hidden neurons per layer.
            number_normalized_inputs: The number of consecutive input tensors, starting from the first one, which are
                                      batch-normalized.
            normalized_output: Is the output normalized to [-1, 1]?
        """
        super(NeuralNetwork, self).__init__()

        NUMBER_OUTPUT_LAYERS = 1

        self.__number_layers = NUMBER_OUTPUT_LAYERS + number_hidden_layers
        self.__number_normalized_inputs = number_normalized_inputs
        self.__normalized_output = normalized_output

        if number_normalized_inputs > 0:
            number_normalized_neurons = sum(numbers_inputs[0:number_normalized_inputs])
            self.__batch_normalization = BatchNorm1d(number_normalized_neurons)

        number_before = sum(numbers_inputs)

        for index in range(self.__number_layers):
            if index < self.__number_layers - 1:
                number_after = number_hidden_neurons_per_layer
            else:
                number_after = number_outputs

            exec("self.__linear_" + str(index) + " = Linear(number_before, number_after)")

            number_before = number_after

    def __call__(self, *inputs: Tensor) -> Tensor:
        """
        Perform a forward propagation on the given inputs.

        Args:
            inputs: The inputs to be forward propagated after concatenation.

        Returns:
            The output resulting from the forward propagation.
        """
        activations = inputs

        batch_size = len(inputs[0])
        if self.__number_normalized_inputs > 0 and batch_size > 1:
            activations_normalized = activations[0:self.__number_normalized_inputs]
            activations_not_normalized = activations[self.__number_normalized_inputs:]

            activations_normalized = torch.cat(activations_normalized, dim=1)
            activations_normalized = self.__batch_normalization(activations_normalized)

            activations = (activations_normalized, *activations_not_normalized)

        activations = torch.cat(activations, dim=-1)

        for index in range(self.__number_layers):
            activations = eval("self.__linear_" + str(index) + "(activations)")
            if index < self.__number_layers - 1:
                activations = eval(ACTIVATION_FUNCTION + "(activations)")
            elif self.__normalized_output:
                activations = tanh(activations)

        return activations


class Actor(NeuralNetwork):
    """
    An actor network of the form a = pi(o), with the vector observation o and vector action a.
    """

    def __init__(self, number_sensors: int, number_motors: int):
        """
        Initialize the actor network.

        Args:
            number_sensors: The number of sensors.
            number_motors: The number of motors.
        """
        number_normalized_inputs = 1 if USE_BATCH_NORMALIZATION else 0

        super(Actor, self).__init__((number_sensors,), number_motors,
                                    ACTOR_NUMBER_HIDDEN_LAYERS, ACTOR_NUMBER_HIDDEN_NEURONS_PER_LAYER,
                                    number_normalized_inputs, True)


class NonlocalCritic(NeuralNetwork):
    """
    A nonlocal critic network of the form Q(o_1, o_2, ..., a_1, a_2, ...), with the vector observations o_1, o_2, ...,
    the vector actions a_1, a_2, ... and the scalar action-value Q.

    The critic is nonlocal, because it is not based on a single observation and action pair, like for example
    (o_1, a_1), which would be associated to a single location in the environment, where the corresponding sensor and
    motor would be located. Instead, the critic is based on multiple such pairs and thus locations, which means that its
    association to the environment has a nonlocal scope.
    """

    def __init__(self, number_locations: int, number_sensors: int, number_motors: int):
        """
        Initialize the nonlocal critic network.

        Args:
            number_locations: The number of locations the nonlocal critic is associated to.
            number_sensors: The number of sensors.
            number_motors: The number of motors.
        """
        NUMBER_OUTPUTS = 1

        numbers_sensors = number_locations * (number_sensors,)
        numbers_motors = number_locations * (number_motors,)

        numbers_inputs = (*numbers_sensors, *numbers_motors)

        number_normalized_inputs = number_locations if USE_BATCH_NORMALIZATION else 0

        super(NonlocalCritic, self).__init__(numbers_inputs, NUMBER_OUTPUTS,
                                             CRITIC_NUMBER_HIDDEN_LAYERS, CRITIC_NUMBER_HIDDEN_NEURONS_PER_LAYER,
                                             number_normalized_inputs, False)
