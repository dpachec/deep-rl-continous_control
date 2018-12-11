# Project Details

We will work with the Reacher environment V02 of Unity ML-Agents, to train 20 virtual robotic arms to follow a moving target. Our agents will store observations of the environment in a continuous vector of 33 values corresponding to position, rotation, velocity, and angular velocities of the arms. Each agent will have four possible actions available at each time step (4 continous variables with values between -1 and 1).

Robots will receive a reward of +.1 for each time step their hand is within the goal area. The goal is to keep the robot's hand at the target location for as long as possible. 

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

The first cell of the Continous_Control.ipynb file will train 20 agents. The weights of the Actor and Critic neural network implementations will be saved into "checkpoint_actor.pth" and "checkpoint_critic.pth".

To observe the behavior of the trained agent, load the weights from the files and run the simulation in the second cell.







