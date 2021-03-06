# Deep Reinforcement Learning Examples

[//]: # (Image References)

[image_trained_agents]: images/trained_agents.gif "Trained Agents"
[image_kernel]: images/set_kernel_in_jupyter.png "Set Kernel in Jupyter"

![Trained Agents][image_trained_agents]

This repository contains tutorials and examples I implemented and worked through as part of Udacity's [Deep Reinforcement Learning Nanodegree](https://www.udacity.com/course/deep-reinforcement-learning-nanodegree--nd893) program.


The tutorials implement various algorithms in reinforcement learning.  All the code is in the latest version of PyTorch (currently version 1.9) and Python 3 (currently version 3.9.6).

* [Dynamic Programming](dynamic-programming): Implement Dynamic Programming algorithms such as Policy Evaluation, Policy Improvement, Policy Iteration, and Value Iteration. 
* [Monte Carlo](monte-carlo): Implement Monte Carlo methods for prediction and control. 
* [Temporal-Difference](temporal-difference): Implement Temporal-Difference methods such as Sarsa, Q-Learning, and Expected Sarsa. 
* [Discretization](discretization): Learn how to discretize continuous state spaces, and solve the Mountain Car environment.
* [Tile Coding](tile-coding): Implement a method for discretizing continuous state spaces that enables better generalization.
* [Deep Q-Network](dqn): Explore how to use a Deep Q-Network (DQN) to navigate a space vehicle without crashing.
* [Hill Climbing](hill-climbing): Use hill climbing with adaptive noise scaling to balance a pole on a moving cart.
* [Cross-Entropy Method](cross-entropy): Use the cross-entropy method to train a car to navigate a steep hill.
* [REINFORCE](reinforce): Learn how to use Monte Carlo Policy Gradients to solve a classic control task.
* **Proximal Policy Optimization**: Explore how to use Proximal Policy Optimization (PPO) to solve a classic reinforcement learning task. (_Coming soon!_)
* **Deep Deterministic Policy Gradients**: Explore how to use Deep Deterministic Policy Gradients (DDPG) with OpenAI Gym environments.
  * [Pendulum](ddpg-pendulum): Use OpenAI Gym's Pendulum environment.
  * [BipedalWalker](ddpg-bipedal): Use OpenAI Gym's BipedalWalker environment.
* [Finance](finance): Train an agent to discover optimal trading strategies.


### Resources

* [Cheatsheet](cheatsheet): Udacity provide [this useful PDF file](cheatsheet/cheatsheet.pdf) with formulae and algorithms that help with understanding reinforcement learning. 

## OpenAI Gym Benchmarks

### Classic Control
- `Acrobot-v1` with [Tile Coding](tile-coding/Tile_Coding_Solution.ipynb) and Q-Learning  
- `Cartpole-v0` with [Hill Climbing](hill-climbing/Hill_Climbing.ipynb) | solved in 13 episodes
- `Cartpole-v0` with [REINFORCE](reinforce/REINFORCE.ipynb) | solved in 691 episodes 
- `MountainCarContinuous-v0` with [Cross-Entropy Method](cross-entropy/CEM.ipynb) | solved in 47 iterations
- `MountainCar-v0` with [Uniform-Grid Discretization](discretization/Discretization_Solution.ipynb) and Q-Learning | solved in <50000 episodes
- `Pendulum-v0` with [Deep Deterministic Policy Gradients (DDPG)](ddpg-pendulum/DDPG.ipynb)

### Box2d
- `BipedalWalker-v2` with [Deep Deterministic Policy Gradients (DDPG)](ddpg-bipedal/DDPG.ipynb)
- `CarRacing-v0` with **Deep Q-Networks (DQN)** | _Coming soon!_
- `LunarLander-v2` with [Deep Q-Networks (DQN)](dqn/solution/Deep_Q_Network_Solution.ipynb) | solved in 1504 episodes

### Toy Text
- `FrozenLake-v0` with [Dynamic Programming](dynamic-programming/Dynamic_Programming_Solution.ipynb)
- `Blackjack-v0` with [Monte Carlo Methods](monte-carlo/Monte_Carlo_Blackjack.ipynb)
- `CliffWalking-v0` with [Temporal-Difference Methods](temporal-difference/Temporal_Difference_Solution.ipynb)

## Dependencies

To set up your python environment to run the code in this repository, follow the instructions below.

1. Create (and activate) a new environment with Python 3.6.

	- __Linux__ or __Mac__: 
	```bash
	conda create --name DRLND python=3.6
	source activate DRLND
	```
	- __Windows__: 
	```bash
	conda create --name DRLND python=3.6 
	activate DRLND
	```
	
2. Follow the instructions in [this repository](https://github.com/openai/gym) to perform a minimal install of OpenAI gym.  
	- Next, install the **classic control** environment group by following the instructions [here](https://github.com/openai/gym#classic-control).
	- Then, install the **box2d** environment group by following the instructions [here](https://github.com/openai/gym#box2d).
	
3. Clone the repository (if you haven't already!), and navigate to the `python/` folder.  Then, install several dependencies.
```bash
git clone https://github.com/udacity/deep-reinforcement-learning.git
cd deep-reinforcement-learning/python
pip install .
```

4. Create an [IPython kernel](http://ipython.readthedocs.io/en/stable/install/kernel_install.html) for the `DRLND` environment.  
```bash
python -m ipykernel install --user --name DRLND --display-name "DRLND"
```

5. Before running code in a notebook, change the kernel to match the `DRLND` environment by using the drop-down `Kernel` menu. 

![Kernel][image_kernel]
