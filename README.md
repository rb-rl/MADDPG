# MADDPG
This repository contains a deep reinforcement learning multi-agent based on a multi-agent deep deterministic policy gradient (=MADDPG) used for two actors playing tennis in a 3D Unity environment.

The MADDPG extends the single-agent [DDPG](https://github.com/rb-rl/DDPG) to multiple agents. In [Comparison.md](https://github.com/rb-rl/MADDPG/blob/main/Comparison.md), this extension is compared to an alternative approach based on a self-playing DDPG agent.

## Environment

The environment is a floor in 3D-space with a tennis court consisting of a net, two rackets and a ball placed on it. It is based on the Unity engine and is provided by Udacity. The continuous observations, continuous actions and the rewards for each agent are given as follows:

**Observation**

- 24 floating point values with following shape:  
  - 3 covered time stamps
  - 8 floating point values per covered time stamp = position and velocity of own racket and ball  

Note that the current time stamp, consisting of 8 floating point values, is written in the Python index range [16:24] of the 24 values observation. In the next timestamp, these 8 floating point values are moved to the range [8:16] and afterwards to [0:8]. Also, in contrast to [DDPG](https://github.com/rb-rl/DDPG/blob/main/README.md), we have observations and not states, because each agent only observes a part of the environment and can therefore not define the whole state of the environment. 

**Action**

- 2 floating point values in [-1,1] = position of own racket

Note that the trivial action value (0, 0) does not lead to a fixed position, but the environment also contains a slight noise on the rackets.

**Reward**

- +0.1  = agent hits ball over net
- -0.01 = agent lets incoming ball hit ground or plays ball out of bounds

The environment is episodic. The return of an agent per episode, which is its non-discounted cumulative reward, is referred to as an individual score. The collective score is the maximum of the individual scores of the two agents. The environment is considered as solved if the collective score averaged over the 100 most recent episodes reaches +0.5.

## Demo

The repository adresses both training and inference of the agent. The training process can be observed in a Unity window, as shown in the following video.

https://user-images.githubusercontent.com/92691697/139115332-5475871a-4917-45e4-972e-b6813504fe51.mp4

When the training is stopped, the actor and critic neural networks of the two agents are stored in the files called agent_0_actor.pt, agent_0_critic.pt, agent_1_actor.pt and agent_1_critic.pt.

The files [agent_0_actor.pt](agent_0_actor.pt), [agent_0_critic.pt](agent_0_critic.pt), [agent_1_actor.pt](agent_1_actor.pt) and [agent_1_critic.pt](agent_1_critic.pt) provided in this repository are the neural networks of a successfully trained multi-agent.

The application of the multi-agent on the environment, i.e. the inference process, can also be observed in a Unity window with this repository:

https://user-images.githubusercontent.com/92691697/139140519-32f17d91-a50f-4ad3-8632-79a7519008ba.mp4

## Installation

In order to install the project provided in this repository, follow these steps:

- Install a 64-bit version of [Anaconda](https://anaconda.cloud/installers)
- Open the Anaconda prompt and execute the following commands:
```
conda create --name drlnd python=3.6
activate drlnd

git clone https://github.com/udacity/deep-reinforcement-learning.git

cd deep-reinforcement-learning/python
```
- Remove `torch==0.4.0` in the file `requirements.txt` located in the current folder `.../python`
- Continue with the following commands:
```
pip install .
pip install keyboard
conda install pytorch=0.4.0 -c pytorch

python -m ipykernel install --user --name drlnd --display-name "drlnd"

cd ..\..

git clone git@github.com:rb-rl/MADDPG.git
cd MADDPG
```
- For Windows users: If you do not know whether you have a 64-bit operating system, you can use this [help](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64)
- Download the Udacity Unity Tennis environment matching your environment:
  - [Linux Version](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip)
  - [Mac OSX Version](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip)
  - [Windows (32-bit) Version](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86.zip)
  - [Windows (64-bit) Version](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip)
  - [Amazon Web Services](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux_NoVis.zip)
- For Amazon Web Services users: You have to deactivate the [virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md) and perform the training in headless mode. For inference, you have to activate the virtual screen and use the Linux Version above
- Unzip the zip file into the folder `MADDPG` such that the `Tennis.exe` in the zip file has the relative path `MADDPG\Tennis_Windows_x86_64\Tennis.exe`
- Start a jupyter notebook with the following command:
```
jupyter notebook
```
- Open `Main.ipynb`
- In the Jupyter notebook, select `Kernel -> Change Kernel -> drlnd`

## Usage

In order to do the training and inference by yourself, simply open [Main.ipynb](Main.ipynb) and successively execute the Jupyter Notebook cells by pressing `Shift+Enter`.
