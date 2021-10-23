# MADDPG
This repository contains a deep reinforcement learning meta agent based on a multi-agent deep deterministic policy gradient (=MADDPG) used for two actors playing tennis in a 3D Unity environment.

The MADDPG extends the single-agent [DDPG](https://github.com/rb-rl/DDPG) to multiple agents.

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
