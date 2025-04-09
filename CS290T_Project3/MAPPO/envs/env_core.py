from pettingzoo.mpe import simple_spread_v3

class EnvCore(object):
    # https://pettingzoo.farama.org/environments/mpe/simple_spread/
    def __init__(self):
        self.env = simple_spread_v3.parallel_env(
            render_mode="rgbarray", 
            N=3, 
            local_ratio=0.5, 
            max_cycles=30, 
            continuous_actions=False)
        
        self.agent_num = 3  
        self.obs_dim = 18
        self.action_dim =5

    def reset(self):
        sub_agent_obs = []
        observations, _ = self.env.reset()
        for agent in self.env.agents:
            sub_agent_obs.append(observations[agent])
        return sub_agent_obs

    def step(self, actions):
        sub_agent_obs = []
        sub_agent_reward = []
        sub_agent_done = []
        sub_agent_info = []

        i = 0
        actions_step = {}
        for agent in self.env.agents:
            if i < len(actions):
                actions_step[agent] = actions[i][0]
                i += 1

        observations, rewards, terminations, truncations, infos = self.env.step(actions_step)
        for agent in self.env.agents:
            sub_agent_obs.append(observations[agent])
            sub_agent_reward.append([rewards[agent]])
            sub_agent_done.append(terminations[agent])
            sub_agent_info.append(infos[agent])

        return [sub_agent_obs, sub_agent_reward, sub_agent_done, sub_agent_info]

