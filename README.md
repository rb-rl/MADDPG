# MADDPG
This repository contains a deep reinforcement learning meta agent based on a multi-agent deep deterministic policy gradient (=MADDPG) used for two actors playing tennis in a 3D Unity environment.

The MADDPG extends the single-agent [DDPG](https://github.com/rb-rl/DDPG) to multiple agents.

## Environment

The environment is a floor in 3D-space with a tennis court consisting of a net, two rackets and a ball placed on it. It is based on the Unity engine and is provided by Udacity. The continuous states, continuous actions and the rewards for each agent are given as follows:

**State**

- 24 floating point values with following shape:  
  - 3 covered time stamps
  - 8 floating point values per covered time stamp = position and velocity of own racket and ball  

Note that the current time stamp, consisting of 8 floating point values, is written in the Python index range [16:24] of the 24 values state. In the next timestamp, these 8 floating point values are moved to the range [8:16] and afterwards to [0:8].

**Action**

- 2 floating point values in [-1,1] = position of own racket

Note that the action [0,0] does not lead to a fixed position, but the environment also contains a slight noise on the rackets.

**Reward**

- +0.1  = agent hits ball over net
- -0.01 = agent lets incoming ball hit ground or plays ball out of bounds

The environment is episodic. The return of an agent per episode, which is its non-discounted cumulative reward, is referred to as an individual score. The collective score is the maximum of the individual scores of the two agents. The environment is considered as solved if the collective score averaged over the 100 most recent episodes reaches +0.5.

## Installation

In order to install the project provided in this repository on Windows 10, follow these steps:

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
- Download the [Udacity Unity Tennis environment](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip)
- Unzip the zip file into the folder `MADDPG` such that the `Tennis.exe` in the zip file has the relative path `MADDPG\Tennis_Windows_x86_64\Tennis.exe`
- Start a jupyter notebook with the following command:
```
jupyter notebook
```
- Open `Main.ipynb`
- In the Jupyter notebook, select `Kernel -> Change Kernel -> drlnd`

## Usage

In order to do the training and inference by yourself, simply open [Main.ipynb](Main.ipynb) and successively execute the Jupyter Notebook cells by pressing `Shift+Enter`.
