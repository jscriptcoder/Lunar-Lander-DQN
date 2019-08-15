# Import the necessary packages
import gym
import torch

from utils import run_gym, train_agent

gym.logger.set_level(40)

# Instantiate the environment
env = gym.make('LunarLander-v2')
env.seed(0)

print('State shape: ', env.observation_space.shape)
print('Number of actions: ', env.action_space.n)

# Let's explore the enviroment with random acitions
#run_gym(env)

from dqn_agent import Agent

state_size = env.observation_space.shape[0]
action_size = env.action_space.n

# Instantiate agent
agent = Agent(state_size=state_size, action_size=action_size, seed=123)

# Let's watch an untrained agent
#run_gym(env, get_action=lambda state: agent.act(state))

#scores = train_agent(agent, env)

agent.qnetwork_local.load_state_dict(torch.load('checkpoint.pth'))

run_gym(env, get_action=lambda state: agent.act(state), max_t=1000)

#import matplotlib.pyplot as plt
#plt.plot(scores)