# Comparison MADDPG vs self-playing DDPG

The tennis playing problem does not necessarily require a multi-agent approach like MADDPG. It is sufficient to have a single DDPG agent which plays against itself. In the following, we have a closer look at the alternative approach of the self-playing DDPG and compare it to MADDPG.

## Demo

The files [agent_actor.pt](agent_actor.pt) and [agent_critic.pt](agent_critic.pt) provided in this repository are the neural networks of a successfully trained self-playing DDPG agent.

The application of the DDPG agent on the environment, i.e. the inference process, can be observed in a Unity window with this repository:

https://user-images.githubusercontent.com/92691697/139150068-55d8be23-a504-4d5e-b69b-975f7298473b.mp4

## Comparison

When comparing [Main.ipynb](Main.ipynb) and [Main_DDPG.ipynb](Main_DDPG.ipynb), we see that the MADDPG multi-agent takes 5044 episodes to solve the environment, whereas the self-playing DDPG agent needs only 2597 episodes and is thus trains significantly faster:

![Episodes_Number](https://user-images.githubusercontent.com/92691697/139137963-b1550f11-4c6b-494a-9efb-51f1e0805c8c.PNG)
![Episodes_Number_DDPG](https://user-images.githubusercontent.com/92691697/139150646-d56b9884-3612-4068-aa17-dd1f450dc32d.PNG)

This can also be seen in the plots of the average collective scores:

![Score](https://user-images.githubusercontent.com/92691697/139138187-b2e1c1f0-3ea5-44eb-a6d1-e8207cc738c2.PNG)
![Score_DDPG](https://user-images.githubusercontent.com/92691697/139150656-06262acc-8a85-4b09-b61d-fae2516e8336.PNG)

## Installation

In order to install the self-playing DDPG agent, simply follow the installation instructions of [DDPG](https://github.com/rb-rl/DDPG/blob/main/README.md) and copy the file [Main_DDPG.ipynb](Main_DDPG.ipynb) provided from the present repository to the DDPG repository.

## Usage

In order to do the training and inference of the self-playing DDPG agent by yourself, simply open [Main_DDPG.ipynb](Main_DDPG.ipynb) and successively execute the Jupyter Notebook cells by pressing `Shift+Enter`.
