# Episodic Model Based Learning using Maximum Likelihood Estimate of the Environment
# Do not change the arguments and output types of any of the functions provided! 
import numpy as np
import gym
import copy

from dynamic_programming import PolicyIteration, ValueIteration, print_agent

import matplotlib.pyplot as plt
import pdb


def initialize_P(nS, nA):
  """Initializes a uniformly random model of the environment with 0 rewards.

    Parameters
    ----------
    nS: int
      Number of states
    nA: int
      Number of actions

    Returns
    -------
    P: np.array of shape [nS x nA x nS x 4] where items are tuples representing transition information
      P[state][action] is a list of (prob, next_state, reward, done) tuples.
  """
  P = [[[(1.0/nS, i, 0, False) for i in range(nS)] for _ in range(nA)] for _ in range(nS)]

  return P

def initialize_counts(nS, nA):
  """Initializes a counts array.

    Parameters
    ----------
    nS: int
      Number of states
    nA: int
      Number of actions

    Returns
    -------
    counts: np.array of shape [nS x nA x nS]
      counts[state][action][next_state] is the number of times that doing "action" at state "state" transitioned to "next_state"
  """
  counts = [[[0 for _ in range(nS)] for _ in range(nA)] for _ in range(nS)]

  return counts

def initialize_rewards(nS, nA):
  """Initializes a rewards array. Values represent running averages.

    Parameters
    ----------
    nS: int
      Number of states
    nA: int
      Number of actions

    Returns
    -------
    rewards: array of shape [nS x nA x nS]
      counts[state][action][next_state] is the running average of rewards of doing "action" at "state" transtioned to "next_state"
  """
  rewards = [[[0 for _ in range (nS)] for _ in range(nA)] for _ in range(nS)]

  return rewards

def counts_and_rewards_to_P(counts, rewards):
  """Converts counts and rewards arrays to a P array consistent with the Gym environment data structure for a model of the environment.
    Use this function to convert your counts and rewards arrays to a P that you can use in value iteration.

    Parameters
    ----------
    counts: array of shape [nS x nA x nS]
      counts[state][action][next_state] is the number of times that doing "action" at state "state" transitioned to "next_state"
    rewards: array of shape [nS x nA x nS]
      counts[state][action][next_state] is the running average of rewards of doing "action" at "state" transtioned to "next_state"

    Returns
    -------
    P: np.array of shape [nS x nA x nS x 4] where items are tuples representing transition information
      P[state][action] is a list of (prob, next_state, reward, done) tuples.
  """
  nS = len(counts)
  nA = len(counts[0])
  P = [[[] for _ in range(nA)] for _ in range(nS)]

  for state in range(nS):
    for action in range(nA):
      if sum(counts[state][action]) != 0:
        for next_state in range(nS):
          if counts[state][action][next_state] != 0:
            prob = float(counts[state][action][next_state]) / float(sum(counts[state][action]))
            reward = rewards[state][action][next_state]
            P[state][action].append((prob, next_state, reward, False))
      else:
        prob = 1.0 / float(nS)
        for next_state in range(nS):
          P[state][action].append((prob, next_state, 0, False))

  return P


def update_mdp_model_with_history(counts, rewards, history):
    """Given a history of an entire episode, update the count and rewards arrays"""
    
    for transition in history:
        state, action, reward, next_state, done = transition
        
        # Update counts
        counts[state][action][next_state] += 1
        
        # Update running average of rewards
        old_avg = rewards[state][action][next_state]
        count = counts[state][action][next_state]
        rewards[state][action][next_state] = old_avg + (reward - old_avg) / count
        
        # For terminal states, ensure probability of returning to itself is 1
        if done:
            # Reset all counts for this state-action pair
            for ns in range(len(counts[state][action])):
                if ns != next_state:
                    counts[state][action][ns] = 0
            # Set count for this transition to 1
            counts[state][action][next_state] = 1
            rewards[state][action][next_state] = reward


def learn_with_mdp_model(env, num_episodes=5000, gamma=0.95, e=0.8, decay_rate=0.99):
    """Build a model of the environment and use value iteration to learn a policy."""
    
    P = initialize_P(env.nS, env.nA)    
    counts = initialize_counts(env.nS, env.nA)
    rewards = initialize_rewards(env.nS, env.nA)

    policy = np.zeros(env.nS, dtype=int)
    All_episodes = np.load("All_episodes.npy", allow_pickle=True)
    # breakpoint()
    pdb.set_trace()
    
    
    # Process all episodes from the loaded data
    for episode in All_episodes:
        update_mdp_model_with_history(counts, rewards, episode)
    
    # Convert counts and rewards to P format
    P = counts_and_rewards_to_P(counts, rewards)
    
    # Set terminal state flags
    # In Frozen Lake, state 15 (bottom-right) is the goal state
    goal_state = env.nS - 1
    for a in range(env.nA):
        for i in range(len(P[goal_state][a])):
            prob, next_state, reward, _ = P[goal_state][a][i]
            P[goal_state][a][i] = (prob, next_state, reward, True)
    
    # Find hole states (states with negative rewards)
    for s in range(env.nS):
        for a in range(env.nA):
            for i in range(len(P[s][a])):
                prob, next_state, reward, done = P[s][a][i]
                if reward < 0:  # If it's a hole
                    P[s][a][i] = (prob, next_state, reward, True)
    
    # Run value iteration to get the optimal policy
    theta = 1e-8  # Convergence threshold
    value_iter = ValueIteration(env.ncol, env.nrow, P, theta, gamma)
    value_iter.value_iteration()
    
    # Extract policy from value iteration
    for s in range(env.nS):
        policy[s] = np.argmax(value_iter.pi[s])
    
    return policy


def render_single(env, policy):
    """Renders policy once on environment. Watch your agent play!
    
    Parameters
    ----------
    env: gym.core.Environment
      Environment to play on. Must have nS, nA, and P as
      attributes.
    Policy: np.array of shape [env.nS]
      The action to take at a given state
    """

    episode_reward = 0
    state = env.reset()
    done = False
    while not done:
        env.render()
        # time.sleep(0.5) # Seconds between frames. Modify as you wish.
        action = policy[state]
        state, reward, done, _ = env.step(action)
        episode_reward += reward

    print("Episode reward: %f" % episode_reward)
    return episode_reward




# Don't change code here !
def main():
    env = gym.make('FrozenLake-v0')
    env = env.unwrapped
    env.np_random.seed(456465)

    policy = learn_with_mdp_model(env, num_episodes=1000)

    score = []
    for i in range(100):
        episode_reward = render_single(env, policy)
        score.append(episode_reward)
    for i in range(len(score)):
        score[i] = np.mean(score[:i + 1])

    plt.plot(np.arange(100), np.array(score))
    plt.title('The running average score of the model based agent')
    plt.xlabel('training episodes')
    plt.ylabel('score')
    plt.savefig('model_based.png')
    plt.show()


if __name__ == '__main__':
    main()
