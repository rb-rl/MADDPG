# Comparison MADDPG vs self-playing DDPG

The tennis playing problem does not necessarily require a multi-agent approach like MADDPG. It is sufficient to have a single DDPG agent which plays against itself. In the following, we have a closer look at the alternative approach of the self-playing DDPG and compare it to MADDPG.

## Demo

The files [agent_actor.pt](agent_actor.pt) and [agent_critic.pt](agent_critic.pt) provided in this repository are the neural networks of a successfully trained self-playing DDPG agent.

The application of the DDPG agent on the environment, i.e. the inference process, can be observed in a Unity window with this repository:

https://user-images.githubusercontent.com/92691697/139150068-55d8be23-a504-4d5e-b69b-975f7298473b.mp4

## Installation

In order to install the self-playing DDPG agent, simply follow the installation instructions of [DDPG](https://github.com/rb-rl/DDPG/blob/main/README.md) and copy the file Main_DDPG.ipynb provided from the present repository to the DDPG repository.

## Usage

In order to do the training and inference of the self-playing DDPG agent by yourself, simply open [Main_DDPG.ipynb](Main_DDPG.ipynb) and successively execute the Jupyter Notebook cells by pressing `Shift+Enter`.
