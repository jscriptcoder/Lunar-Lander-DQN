# Import the necessary packages
from utils import make_env, run_gym, train_agent, plot_scores

env = make_env('LunarLander-v2')

print('State shape: ', env.observation_space.shape)
print('Number of actions: ', env.action_space.n)

# Let's explore the enviroment with random acitions
#run_gym(env)

from agent import DQNAgent

state_size = env.observation_space.shape[0]
action_size = env.action_space.n

# Instantiate agent
agent = DQNAgent(state_size=state_size, action_size=action_size, 
#                 use_double=True, 
#                 use_dueling=True, 
                 use_priority=True,
#                 use_noise=True,
                 seed=0
                )

agent.summary()

# Let's watch an untrained agent
#run_gym(env, get_action=lambda state: agent.act(state))

scores = train_agent(agent, env)

plot_scores(scores, 'Prioritized  Deep Q-Network', polyfit_deg=6)

#agent.load_weights('prioritized_local_weights.pth')

run_gym(env, get_action=lambda state: agent.act(state), max_t=1000)
