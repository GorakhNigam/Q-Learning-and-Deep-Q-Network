# Q-Learning-and-Deep-Q-Network
In this project, we will implement Q-Learning and Deep Q-Network (DQN) algorithms to optimize the behavior of a reinforcement learning (RL) agent applied to a maze pathfinding problem. Our task includes designing and tuning these algorithms to enable effective agent navigation.

## Q-Learning (Tabular)
	•	Maintains a Q-table of state–action values
	•	Uses the Bellman optimality update rule
	•	Suitable for small, discrete state spaces
  
## Deep Q-Network (DQN)
	•	Replaces the Q-table with a neural network
	•	Uses experience replay to decorrelate training samples
	•	Uses a target network for stable learning
	•	Implemented using PyTorch

## Action Selection Strategy
Both agents use an ε-greedy policy:    
	•	With probability ε, select a random action (exploration)     
	•	With probability 1 − ε, select the action with the highest Q-value (exploitation)     

ε is decayed gradually over episodes to shift from exploration to exploitation.     

## Experimental Setup
	•	Maze size: 5 × 5
	•	Training episodes: 500
	•	Maximum steps per episode enforced
	•	Reward tracked per episode
	•	Visualization enabled via Pygame

## Results

The agent shows clear learning behavior:   
	•	Early episodes exhibit large negative rewards due to random exploration    
	•	Rewards improve steadily as the agent learns valid paths     
	•	Final episodes demonstrate stable, near-optimal navigation behavior     
