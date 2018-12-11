# Project Details

We will work with the Reacher environment V02 of Unity ML-Agents, to train a (virtual) robotic arm to follow a moving target. Our agent will store observations of the environment in a continuous vector of 33 values corresponding to position, rotation, velocity, and angular velocities of the arm. It will have four possible actions available at each time step (4 continous variable with values between -1 and 1).

The robot will receive a reward of +.1 for each step its hand is in the goal location. Thus its goal is to maintain the position of the arm position at the target location for as long as possible. 

20 instances of the robot will work in parallell in the same environment.  
The environment is considered solved when an average score of +30 is achieved across all agents and 100 episodes.  


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







