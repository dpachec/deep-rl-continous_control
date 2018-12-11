# Report

## Introduction
Description of a Deep Deterministic Policy Gradient (DPDG) algorithm developed to solve Unity ML-agents Reacher V02 environment.

Use the Continuous_Control.ipynb to train the agent. 
The code in the first cell organizes the interaction with the Unity environment, collects trajectories and run the ddpg algorithm in the ddpg_agent.py file.

For each episode during training:

• Each agent collects an observation from the environment and takes an action using the policy, observes the reward and the next state.

• Stores the experience tuple SARS' in replay memory.

• Select a small bunch of tuples from memory randomly and learn from it ddpq_agent.py

## Deep deterministic Policy Gradient Agent
ddpg_agent.py has 3 main classes: 

- **Agent**, with parameters state_size, action_size, and a seed for random number generation in PyTorch.
- **ReplayBuffer**, initialized with parameters action_size, BUFFER_SIZE, BATCH_SIZE, and seed.
- **OUNoise** with inputs size, seed, mu, theta, sigma.


### Agent
Four neural networks are initialized with the Agent. Basically, two networks with two instances: an Actor and a Critic network, with two versions (target and local). The critic network estimates the value function of policy pi, V(pi) using the TD estimate. The output of the critic is used to train the actor network, which takes in a state, and outputs a distribution over possible actions.

The algorithm goes like this:
- Input the current state into the actor and get the action to take in that state. Observe next state and reward, to get an experience tuple s a r s'
- Then, using the TD estimate, which is the reward R plus the critic's estimate for s', so r+yV(s';Thetav), the critic network is trained.
- Next, to calculate the advantage r+yV(s') - V(s), we also use the critic network.
- Finally, the actor is trained using the calculated advantage as a baseline.

This type of architecture is well suited for continuous action spaces, with the critic network learning the optimal action for every given state, and then passing that optimal action to the actor network which uses it to estimate the policy.
We use two networks for each of the two to control the update of the weights. Only when the function **learn2()** is called after each episode the weights are transferred between the target and local versions.

### Main functions

**step(self, state, action, reward, next_state, done):** Saves the experiences of 20 agents in the replay memory buffer

**act (self, state, eps):**
Selects an action for a given state following the policy encoded by the NN function approximator. The architecture of this network is defined in the model.py file. Main steps:
1) Transformation of the current state from numpy to torch is performed. 
2) Forward pass on the actor_local. 
3) Data is moved to a cpu and to numpy
4) Noise is added
5) Clipping between -1 and 1 .

**learn2(self, experiences, gamma):**
Sample a bunch of experiences across all agents and learn from them by calling the learn function.

**learn(self, experiences, gamma):**
Updates the Actor and Critic network’s weights given a batch of experience tuples.
1) In the update critic section of the function, it first gets the max predicted Q values (for next states) from the critic_target and actor_target models, and compute Q targets for current states. Then, it gets the expected Q values from the critic_local model, compute the loss and minimize the loss using gradient descent. Note that we use gradient clipping.

2) In the update actor section, we compute the loss of he acor, and get the predicted actions.
3) The function soft_update is called in the end to update the target networks.

**soft update (local_model, target_model, tau):**
Grabs all of the target_model and the local_model parameters (in the zip command), and copies a mixture of both (defined by Tau) into the target_param.
The target network receives updates that are a combination of the local (most up to date network) and the target (itself). In general, it will be a very large chunk of itself, and a very small chunk of the other network.

### Replay Buffer
The replay buffer class retains the end most recent experience tuples. If not enough experience is available to the agent (i.e., if self.memory < BATCH_SIZE), no learning takes place.
Buffer is implemented with a python deque. Note that we do not clear out the memory after each episode, which enables to recall and build batches of experience from across episodes.
Given that maxlen is specified to BATCH_SIZE, the buffer is bounded. Once full, when new items are added, a corresponding number of items are discarded from the opposite end.

### OUNoise
Implementation of stochastic noise.


### Neural Network Architecture
Two classes are instantiated in the model.py file for Actor and Critic networks.

**The actor network:**
Neural Network that estimates the optimal policy. 
Built in PyTorch (nn package). 
Architecture includes an input layer (of size = state size), two fully connected hidden layers of 400 and 300 units and an output layer (size = action size).
RELU (Regularized linear units) activation is applied in the forward function for the first two layers. Note that on the output side, given the continuous space we use the Tahn activation function.

**The critic network:**
Approximates the value function. It is defined by subclassing torch.nn.Module. 
The architecture includes an input layer (of size = state size), one fully connected hidden layer of 400 units, a second hidden layer of 300 + action size units and an output layer (size = action size).
RELU activation is applied in the forward function. Note that in the forward pass, the action is being concatenated to get the value of the specific state-action pair.


## Chosen hyperparameters

- BUFFER_SIZE     =  int(5e5)   #replay buffer size
- BATCH_SIZE       =  512 #minibatch size
- GAMMA               =  0.99 #discount factor
- TAU                      =  1e-3 #for soft update of target parameters
- LR_ACTOR          =  1e-3 #learning rate of the actor
- LR_CRITIC           =  3e-3 #learning rate of the critic 
- WEIGHT_DECAY  =  0 #L2 weight decay



## Training protocol
The agent was trained until an average score of +30 for all 20 agents, over 100 consecutive episodes was reached. This was achieved after 442 episodes:


![alt text](/output2.png?raw=true "Title")

Figure 1. Figure shows that average reward over 100 episodes and over all 20 agents. Red dotted line indicates average of +30.


## Ideas for future work
- Try other neural network architectures such as PPO or AC2. 
- Train the agent using raw pixels.


