
import numpy as np
import gym ### 
from gym import spaces

class RPSEnv(gym.Env):
    # toy env for testing sanity of the algorithm
    def __init__(self, max_episode_steps=1, **kargs):
        self.opponent = None
        self.train = False
        # Observation is a Cartesian space of the agent's and the opponent's energy,
        # as well as the current game state (ongoing 0/lose 1/win 2)
        self.win_state_id = 1
        self.loss_state_id = 2
        self.n_ternimal = 2
        self.observation_space = spaces.Discrete(3)

        # Action space is the maximum number of actions possible
        self.action_space = spaces.Discrete(3)

        self.max_episode_steps = max_episode_steps
        self.action_matrix = np.array([self.available_actions(s) for s in range(self.observation_space.n)])

    def _get_info(self):
        return {
            "agent_action":self._agent_action,
            "opponent_action":self._opponent_action,
            "game_state":self._game_state
            }

    def reset(self, seed=None, opponent=None, **kargs):
        super().reset(seed=seed)
        if opponent is not None:
            self.opponent = opponent
        # game start
        self._game_state=0

        # value init
        self._agent_action = None
        self._opponent_action = None

        observation = self._game_state
        info = self._get_info()
        self._i_step = 0
        return observation, info
    
    def available_actions(self, state):
        return np.ones(3)

    def step(self, action):
        self._agent_action = action
        if self.opponent is not None:
            self._opponent_action = self.opponent.step(self._game_state, self.action_space.n)
        else:
            self._opponent_action = self.action_space.sample()
        
        d=self._agent_action-self._opponent_action
        if d%3==1:
            self._game_state = 1
            reward = 1

        elif d%3==2:
            self._game_state = 2
            reward = -1

        else:
            self._game_state=0
            reward = 0

        # An episode is done iff one agent has won
        terminated = self._game_state>0
        observation = self._game_state
        info = self._get_info()

        self._i_step+=1
        truncated = self._i_step>=self.max_episode_steps

        return observation, reward, terminated, truncated, info

