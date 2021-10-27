# ######################################################################################################################
# A reinforcement learning agent.
#
# Code adaption of [1].
#
# [1] https://github.com/rb-rl/DDPG/network.py.
#
# ######################################################################################################################

import network
import memory

if network.FAST_DEVELOPMENT_MODE:
    import importlib

    importlib.reload(network)
    importlib.reload(memory)
    print("Fast development reload: network")
    print("Fast development reload: memory")

from network import Actor, NonlocalCritic, NeuralNetwork
from memory import ReplayMemory

import torch

import numpy as np
import torch.nn.functional as F

from numpy import ndarray
from pathlib import Path
from random import randint, random
from torch import Tensor
from torch.optim import Adam
from typing import List, Tuple

# GENERAL --------------------------------------------------------------------------------------------------------------

# The used device in {"cuda", "cpu"}.
#
# Note that if you have a GPU which requires at least CUDA 9.0, the usage of the CPU is recommended, because otherwise
# the execution might be unexpectedly slow.
DEVICE = "cpu"

# LEARNING -------------------------------------------------------------------------------------------------------------

# The learning rate.
LEARNING_RATE = 0.001

# The interval the epsilon value of epsilon greediness may be in.
EPSILON_INTERVAL = [0.2, 0.2]

# The amount of epsilon decay.
EPSILON_DECAY = 1

# Decay rate of the Ornstein-Uhlenbeck noise in [0,1], see [1], where 0 full noise conservation and 1 maximum noise
# decay.
#
# [1] en.wikipedia.org/wiki/Ornstein–Uhlenbeck_process
THETA = 0.2

# The discount factor.
GAMMA = 0.99

# The batch size.
BATCH_SIZE = 64

# The loss function.
LOSS = "mse_loss"

# The soft update rate of target deep Q-network.
TAU = 0.01

# The number of frames per update of the target deep Q-network.
FRAMES_PER_UPDATE = 4


class Agent:
    """
    A single agent of a multi-agent based on multi-agent deep deterministic gradient descent [1].

    [1] Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments, 2017, arxiv.org/pdf/1706.02275.pdf
    """

    def __init__(self, number_agents: int, number_sensors: int, number_motors: int, agents: List, index: int):
        """
        Initialize the agent.

        Args:
            number_agents: The number of agents in the multi-agent this agent belongs to.
            number_sensors: The number of sensors.
            number_motors: The number of motors.
            agents: The multi-agent.
            index: The index of the agent in the multi-agent.
        """
        self.__number_agents = number_agents
        self.__number_motors = number_motors
        self.__agents = agents
        self.__index = index

        device_name = DEVICE if torch.cuda.is_available() else "cpu"
        print("Used device:", device_name)

        print()

        self.__device = torch.device(device_name)

        self.__policy_network = Actor(number_sensors, number_motors).to(self.__device)
        self.__policy_network_target = Actor(number_sensors, number_motors).to(self.__device)

        self.__q_network = NonlocalCritic(number_agents, number_sensors, number_motors).to(self.__device)
        self.__q_network_target = NonlocalCritic(number_agents, number_sensors, number_motors).to(self.__device)

        print("Policy Network -", self.__policy_network)
        print()
        print("Target Policy Network - Target", self.__policy_network_target)
        print()

        print("Q Network -", self.__q_network)
        print()
        print("Target Q Network - Target", self.__q_network_target)
        print()

        self.__policy_network_optimizer = Adam(self.__policy_network.parameters(), lr=LEARNING_RATE)
        self.__q_network_optimizer = Adam(self.__q_network.parameters(), lr=LEARNING_RATE)

        self.__replay_memory = ReplayMemory(self.__device)

        self.__epsilon = EPSILON_INTERVAL[1]
        self.__noise = self.__epsilon * np.random.randn(number_motors)

        self.__step = 0

    @property
    def epsilon(self) -> float:
        """
        The getter for the epsilon value.

        Returns:
            The epsilon value.
        """
        return self.__epsilon

    def __call__(self, observation: ndarray) -> ndarray:
        """
        Let the agent act on the given observation based on the policy network and an Ornstein-Uhlenbeck noise.

        See equation (7) on page 4 and Algorithm 1 on page 5 of [1].

        [1] Continuous control with deep reinforcement learning, 2015, arxiv.org/pdf/1509.02971.pdf

        Args:
            observation: The current observation of the agent.

        Returns:
            The selected action in [-1,1].
        """
        input = torch.from_numpy(observation).float().unsqueeze(0).to(self.__device)

        self.__policy_network.eval()

        with torch.no_grad():
            output = self.__policy_network(input)

        self.__policy_network.train()

        action = np.clip(output[0].cpu().data.numpy() + self.__noise, -1, 1)

        return action

    def learn(self, observations: ndarray, actions: ndarray, rewards: List[float], next_observations: ndarray,
              dones: List[bool]):
        """
        Perform a learning step.

        Args:
            observations: The current observations.
            actions: The actions taken in the current observations, where every component is in [-1, 1].
            rewards: The rewards obtained by going from the current to the next observations.
            next_observations: The next observations.
            dones: Is the episode done?
        """
        self.__replay_memory.add(observations, actions, rewards, next_observations, dones)

        batch_size = min(BATCH_SIZE, len(self.__replay_memory))
        experiences = self.__replay_memory.extract_random_experiences(batch_size)

        self.__update_policy_network(experiences[0], experiences[1])
        self.__update_q_network(experiences)

        self.__step += 1
        if self.__step % FRAMES_PER_UPDATE == 0:
            self.__soft_update_all()

        self.__epsilon_decay()
        self.__update_noise()

    def save(self, path: str):
        """
        Save the neural networks of the agent.

        Args:
            path: The path to the files where the neural networks should be stored, excluded the file suffix and ending.
        """
        index_text = "_" + str(self.__index)

        actor_path = Path(path + index_text + "_actor").with_suffix(".pt")
        critic_path = Path(path + index_text + "_critic").with_suffix(".pt")

        torch.save(self.__policy_network.state_dict(), actor_path)
        torch.save(self.__q_network.state_dict(), critic_path)

        print(f"Agent saved in ({actor_path}, {critic_path})")

    def load(self, path: str):
        """
        Load the neural networks of the agent.

        Note that the loading is asymmetric to the saving, for simplicity, because we do not save the epsilon value.
        Hence, it is not possible to continue training from a loaded model.

        Args:
            path: The path to the files where the neural networks should be loaded from, excluded the file suffix and
                  ending.
        """
        index_text = "_" + str(self.__index)

        critic_path = Path(path + index_text + "_critic").with_suffix(".pt")
        actor_path = Path(path + index_text + "_actor").with_suffix(".pt")

        self.__q_network.load_state_dict(torch.load(critic_path))
        self.__policy_network.load_state_dict(torch.load(actor_path))

        self.__epsilon = 0

        print(f"Agent loaded from ({actor_path}, {critic_path})")

    def __epsilon_decay(self):
        """
        Perform epsilon decay.
        """
        self.__epsilon = max(EPSILON_INTERVAL[0], EPSILON_DECAY * self.__epsilon)

    def __update_noise(self):
        """
        Update the noise according to the Ornstein-Uhlenbeck process, see [1]:

            noise <- (1 - theta) * noise + epsilon * normal_distribution

        where we use the current epsilon as the standard deviation in this process.

        [1] en.wikipedia.org/wiki/Ornstein–Uhlenbeck_process
        """
        self.__noise = (1 - THETA) * self.__noise + self.__epsilon * np.random.randn(self.__number_motors)

    def __agents_batch(self, batch_agents: Tensor) -> List[Tensor]:
        """
        Flip the batch and agents indices.

        Args:
            batch_agents: A tensor with the indices (batch sample, agent index).

        Returns:
            A tensor list with the indices (agent index, batch sample).
        """
        return [batch_agents[:, index, :] for index in range(self.__number_agents)]

    def __update_policy_network(self, observations_batch_agents: Tensor, actions_batch_agents: Tensor):
        """
        Update the policy network.

        See algorithm 1 on page 13 of  [1]: L = -Q_i(o_1, o_2, ..., a_1, ..., a_{i-1}, pi_i(o_i), a_{i+1}, ...)/N,

                                                with i = self.__index

        Note that a minus occurs above, because a loss is minimized and Q-values are maximized. N is the batch size.

        [1] Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments, 2017, arxiv.org/pdf/1706.02275.pdf

        Args:
            observations_batch_agents: A 2-dimensional set of observations, where the first index refers to the batch
                                       and the second one the agents.
            actions_batch_agents: A 2-dimensional set of actions, where the first index refers to the batch and the
                                  second one the agents.
        """
        observations_agents_batch = self.__agents_batch(observations_batch_agents)
        actions_agents_batch = self.__agents_batch(actions_batch_agents)

        # (a_1, ..., pi_i(o_i), ...)
        actions_agents_batch = [agent.__policy_network(observations_agents_batch[index]) if index == self.__index
                                else actions_agents_batch[index].detach()
                                for index, agent in enumerate(self.__agents)]

        # Q_i(o_1, o_2, ..., a_1, ..., a_{i-1}, pi_i(o_i), a_{i+1}, ...), with i = self.__index
        q_values = self.__q_network(*observations_agents_batch, *actions_agents_batch)

        loss = -q_values.mean()

        self.__policy_network_optimizer.zero_grad()
        loss.backward()

        self.__policy_network_optimizer.step()

    def __update_q_network(self, experiences: Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]):
        """
        Update the deep Q-network.

        See algorithm 1 on page 13 of  [1]: L = LOSS(  r' + gamma * Q'_i(o_1', o_2', ..., pi_1'(o_1'), pi_2'(o_2'), ...)
                                                     - Q_i(o_1, o_2, ..., a_1, a_2, ...)), with i = self.__index

        [1] Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments, 2017, arxiv.org/pdf/1706.02275.pdf

        Args:
            experiences: The experiences used for the update.
        """
        observations_batch_agents, actions_batch_agents, rewards_batch_agents, \
        next_observations_batch_agents, dones_batch_agents = experiences

        observations_agents_batch = self.__agents_batch(observations_batch_agents)
        actions_agents_batch = self.__agents_batch(actions_batch_agents)
        next_observations_agents_batch = self.__agents_batch(next_observations_batch_agents)

        adjusted_rewards = rewards_batch_agents[:, self.__index].unsqueeze(1)
        adjusted_dones = dones_batch_agents[:, self.__index].unsqueeze(1).float()

        # (pi_1'(o_1'), pi_2'(o_2'), ...)
        next_actions_agents_batch = [agent.__policy_network_target(next_observations_agents_batch[index]).detach()
                                     for index, agent in enumerate(self.__agents)]

        # Q_i'(o_1', o_2', ..., pi_1'(o_1'), pi_2'(o_2'), ...), with i = self.__index
        target_q_values = self.__q_network_target(*next_observations_agents_batch, *next_actions_agents_batch).detach()

        # r'+gamma*Q_i'(o_1', o_2', ..., pi_1'(o_1'), pi_2'(o_2'), ...)
        targets = (adjusted_rewards + (GAMMA * target_q_values * (1 - adjusted_dones))).detach()

        # Q_i(o_1, o_2, ...,a_1, a_2, ...)
        q_values = self.__q_network(*observations_agents_batch, *actions_agents_batch)

        loss = eval("F." + LOSS)(q_values, targets)

        self.__q_network_optimizer.zero_grad()
        loss.backward()

        self.__q_network_optimizer.step()

    @staticmethod
    def __soft_update(neural_network: NeuralNetwork, neural_network_target: NeuralNetwork):
        """
        Perform a soft update of a target neural network.

        Args:
            neural_network: The neural network used for soft-updating.
            neural_network_target: The target neural network to be soft-updated.
        """
        for parameters, parameters_target in zip(neural_network.parameters(), neural_network_target.parameters()):
            parameters_target.data.copy_((1 - TAU) * parameters_target.data + TAU * parameters.data)

    def __soft_update_all(self):
        """
        Perform a soft update of the target policy networks and deep Q-networks.
        """
        Agent.__soft_update(self.__policy_network, self.__policy_network_target)
        Agent.__soft_update(self.__q_network, self.__q_network_target)
