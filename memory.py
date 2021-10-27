# ######################################################################################################################
# A replay memory.
#
# Code adaption of [1].
#
# [1] https://github.com/rb-rl/DDPG/network.py.
#
# ######################################################################################################################

import random
import torch
import sys

import numpy as np

from collections import deque
from numpy import ndarray
from torch import Tensor
from typing import List, Tuple, Union

# MEMORY ---------------------------------------------------------------------------------------------------------------

# The number of experiences the replay memory can maximally store.
REPLAY_MEMORY_SIZE = 100000


class ReplayMemory:
    """
    A simple, non-prioritized replay memory.

    The replay memory is optimized for GPU-computations by moving the experiences on the device as tensors.
    """

    def __init__(self, device: torch.device):
        """
        Intialize the replay memory.

        Args:
            device: The used processing unit.
        """
        self.__device = device

        self.__experiences = deque(maxlen=REPLAY_MEMORY_SIZE)

    def add(self, observations: ndarray, actions: ndarray, rewards: List[float], next_observations: ndarray,
            dones: List[bool]):
        """
        Add an experience to the replay memory.

        Args:
            observations: The current observations.
            actions: The actions taken in the current observations, where every component is in [-1, 1].
            rewards: The rewards obtained by going from the current to the next observations.
            next_observations: The next observations.
            dones: Is the episode done?
        """
        experience = tuple(map(self.__to_device, (observations, actions, rewards, next_observations, dones)))
        self.__experiences.append(experience)

    def extract_random_experiences(self, batch_size: int) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """
        Extract a number of random experiences from the replay memory.

        Args:
            batch_size: The number of experiences to be extracted.

        Returns:
            The extracted experiences.
        """
        experiences = random.sample(self.__experiences, batch_size)

        extract_stack = lambda index: torch.stack([experience[index] for experience in experiences])

        return extract_stack(0), extract_stack(1), extract_stack(2), extract_stack(3), extract_stack(4)

    def __to_device(self, variable: Union[List[bool], List[float], ndarray]) -> Tensor:
        """
        Convert a variable to a tensor on the device of the replay memory.

        Args:
            variable: The variable to be converted.

        Returns:
            The variable on the device as a tensor.
        """
        if type(variable) == list:
            if type(variable[0]) == bool:
                variable = np.array(variable, np.uint8)
            else:
                variable = np.array(variable, np.float32)
            tensor = torch.from_numpy(variable)
        elif type(variable) == ndarray:
            tensor = torch.from_numpy(variable).float()
        else:
            print("Not implemented type.")
            sys.exit()

        return tensor.to(self.__device)

    def __len__(self):
        """
        Get the number of experiences in the replay memory.

        Returns:
            The number of experiences.
        """
        return len(self.__experiences)
