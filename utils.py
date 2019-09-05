import gym
import numpy as np
import matplotlib.pyplot as plt

from gym.wrappers import Monitor
from collections import deque

gym.logger.set_level(40)


def make_env(env_id, 
             use_monitor=False, 
             monitor_dir='recordings', 
             seed=42):
    """Instantiates the OpenAI Gym environment
    
    Args:
        env_id (string): OpenAI Gym environment ID
        use_monitor (bool): whether or not to use gym.wrappers.Monitor
        seed (int)
    """
    
    env = gym.make(env_id) # instantiate the environment
    
    if use_monitor: 
        env = Monitor(env, monitor_dir)
        
    env.seed(seed)
    
    return env
    

def run_gym(env, get_action=None, max_t=200):
    """Runs an environment given against actions
    
    Args:
        env (Environment): OpenAI Gym environment https://gym.openai.com/envs
        get_action (func): returns actions based on a state
        max_t (int): maximum number of timesteps
    """
    
    if get_action is None:
        get_action = lambda _: env.action_space.sample()
        
    state = env.reset()
    env.render()
    
    for i in range(max_t):
        action = get_action(state)
        state, reward, done, _ = env.step(action)
        env.render()
    
        if done: break
         
    env.close()



def train_agent(agent, env, 
                n_episodes=2000, max_t=1000, 
                eps_start=1.0, eps_end=0.01, eps_decay=0.995):
    """Deep Q-Learning training
    
    Args:
        agent (DQNAgent): Deep Q-Network agent
        env (Environment): OpenAI Gym environment https://gym.openai.com/envs
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
        eps_start (float): starting value of epsilon, 
            for epsilon-greedy action selection
        eps_end (float): minimum value of epsilon
        eps_decay (float): multiplicative factor (per episode) 
            for decreasing epsilon
    """
    
    scores = [] # list containing scores from each episode
    scores_window = deque(maxlen=100) # last 100 scores
    eps = eps_start # initialize epsilon
    
    for i_episode in range(1, n_episodes+1):
        state = env.reset()
        score = 0
        for t in range(max_t):
            action = agent.act(state, eps)
            next_state, reward, done, _ = env.step(action)
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            
            if done: break
        
        # Save most recent score
        scores_window.append(score)
        scores.append(score)
        
        eps = max(eps_end, eps_decay*eps) # decrease epsilon
        
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(
                i_episode, np.mean(scores_window)), end='')
        
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(
                    i_episode, np.mean(scores_window)))
            
        if np.mean(scores_window)>=200.0:
            print('\nEnvironment solved in {:d} episodes!'.format(i_episode-100))
            print('\nAverage Score: {:.2f}'.format(np.mean(scores_window)))
            
            agent.save_weights()
            
            break
        
    return scores


def plot_scores(scores, title='Deep Q-Network', figsize=(15, 6), polyfit_deg=None):
    fig, ax = plt.subplots(figsize=figsize)
    plt.plot(scores)
    
    if polyfit_deg is not None:
        x = list(range(len(scores)))
        degs = np.polyfit(x, scores, polyfit_deg)
        p = np.poly1d(degs)
        plt.plot(p(x), linewidth=3)
    
    plt.title(title)
    ax.set_ylabel('Score')
    ax.set_xlabel('Epochs')