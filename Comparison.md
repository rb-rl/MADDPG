# Comparison MADDPG vs self-playing DDPG

The tennis playing problem does not necessarily require a multi-agent approach like MADDPG. It is sufficient to have a single DDPG agent which plays against itself. In the following, we have a closer look at the alternative approach of the self-playing DDPG and compare it to MADDPG.

## Demo

The files [agent_actor.pt](agent_actor.pt) and [agent_critic.pt](agent_critic.pt) provided in this repository are the neural networks of a successfully trained self-playing DDPG agent.

The application of the DDPG agent on the environment, i.e. the inference process, can be observed in a Unity window with this repository:

https://user-images.githubusercontent.com/92691697/139150068-55d8be23-a504-4d5e-b69b-975f7298473b.mp4

## Comparison

When comparing [Main.ipynb](Main.ipynb) and [Main_DDPG.ipynb](Main_DDPG.ipynb), we see that the MADDPG multi-agent takes 5044 episodes to solve the environment, whereas the self-playing DDPG agent needs only 2597 episodes and is thus trains significantly faster:

![Episode_Number](https://user-images.githubusercontent.com/92691697/139151306-55bcce5f-018b-481a-9555-90f99b4f9d7c.PNG)
![Episode_Number_DDPG](https://user-images.githubusercontent.com/92691697/139151316-d7d45fe0-4b35-446d-a2d5-4e651108052b.PNG)

This can also be seen in the plots of the average collective scores (upper=MADDPG, lower=self-playing DDPG):

![Score](https://user-images.githubusercontent.com/92691697/139151492-02e9e30c-76da-44a3-b090-d700e7363aa8.PNG)
![Score_DDPG](https://user-images.githubusercontent.com/92691697/139151499-85c75423-6acb-4c91-a32b-c092e847109a.PNG)

The reason for the better training performance is, of course, the symmetry of the environment.

## Installation

In order to install the self-playing DDPG agent, simply follow the installation instructions of [DDPG](https://github.com/rb-rl/DDPG/blob/main/README.md) and copy the file [Main_DDPG.ipynb](Main_DDPG.ipynb) provided from the present repository to the DDPG repository.

## Usage

In order to do the training and inference of the self-playing DDPG agent by yourself, simply open [Main_DDPG.ipynb](Main_DDPG.ipynb) and successively execute the Jupyter Notebook cells by pressing `Shift+Enter`.
