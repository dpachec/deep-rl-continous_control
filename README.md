# Project Details

We will work with the Reacher environment V02 of Unity ML-Agents, to train 20 (virtual) robotic arm to follow a moving target in parallel. Our agents will store observations of the environment in a continuous vector of 33 values corresponding to position, rotation, velocity, and angular velocities of the arms. Each agent will have four possible actions available at each time step (4 continous variable with values between -1 and 1).

Robots will receive a reward of +.1 for each step its hand is in the goal location. Thus the goal is to maintain the arm at the target location for as long as possible. 

The environment is solved when an average score of +30 is achieved across all agents and 100 episodes.  


# Getting Started

The project requires Python 3.6 or higher with the libraries unityagents, numpy, PyTorch.

You will also need the Unity Reacher environment V02, which can be downloaded [here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/.../Reacher.app.zip).


# Installation
1) Clone the repo (Anaconda):
```
git clone https://github.com/dpachec/deep-rl-continous-control.git
```

2) Install Dependencies using Anaconda
```
conda create --name drlnd_continous_control python=3.6
source activate drlnd_continous_control
conda install -y python.app
conda install -y pytorch -c pytorch
pip install unityagents
```

# Instructions

To train the agent use the continous_control.ipynb file. The firs cell will train the agent using the Deep Deterministic Policy Gradient algorithm. The weights of the nn implementation (policy) will be saved. 

To observe the behavior of the trained agent, cell N2, which will load the weights from the files "checkpoint_actor.pth" and "checkpoint_critic.pth". 







