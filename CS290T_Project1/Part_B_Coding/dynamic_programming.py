import copy


class CliffWalkingEnv:
    """ 悬崖漫步环境"""
    def __init__(self, ncol=12, nrow=4):
        self.ncol = ncol  # Define the columns of the grid world
        self.nrow = nrow  # Define the rows of the grid world
        # Transition Matrix P[state][action] = [(p, next_state, reward, done)] Include next states and rewards
        self.P = self.createP()

    def createP(self):
        # Initialization
        P = [[[] for j in range(4)] for i in range(self.nrow * self.ncol)]
        # 4 actions, change[0]:up,change[1]:down, change[2]:left, change[3]:right  
        # Origin of coordinate system:(0,0)   Defined in the upper left corner
        change = [[0, -1], [0, 1], [-1, 0], [1, 0]]
        for i in range(self.nrow):
            for j in range(self.ncol):
                for a in range(4):
                    # Positioned on a cliff or in a target state, any action reward is 0 as interaction cannot continue
                    if i == self.nrow - 1 and j > 0:
                        P[i * self.ncol + j][a] = [(1, i * self.ncol + j, 0,
                                                    True)]
                        continue
                    # Other positions
                    next_x = min(self.ncol - 1, max(0, j + change[a][0]))
                    next_y = min(self.nrow - 1, max(0, i + change[a][1]))
                    next_state = next_y * self.ncol + next_x
                    reward = -1
                    done = False
                    # The next position is on the cliff or at the finish line
                    if next_y == self.nrow - 1 and next_x > 0:
                        done = True
                        if next_x != self.ncol - 1:  # The next position is on the cliff
                            reward = -100
                    P[i * self.ncol + j][a] = [(1, next_state, reward, done)]
        return P    




def print_agent(agent, action_meaning, disaster=[], end=[]):
    print("State Value:")
    value_str = ""
    for i in range(agent.nrow):
        for j in range(agent.ncol):
            value_str += '%6.6s' % ('%.3f' % agent.v[i * agent.ncol + j]) + ' '
        value_str += '\n'  
    print(value_str)

    print("Policy:")    
    policy_str = ""
    for i in range(agent.nrow):
        for j in range(agent.ncol):
            # 一些特殊状态，比如走到悬崖上
            if (i * agent.ncol + j) in disaster:
                policy_str += '💀    '
            elif (i * agent.ncol + j) in end:  # 目标状态
                policy_str += '🏁    '
            else:
                a = agent.pi[i * agent.ncol + j]
                best_a = a.index(max(a))
                if best_a == 0:  # 上
                    policy_str += '↑     '
                elif best_a == 1:  # 下
                    policy_str += '↓     '
                elif best_a == 2:  # 左
                    policy_str += '←     '
                elif best_a == 3:  # 右
                    policy_str += '→     '
        policy_str += '\n'
    print(policy_str)

    with open('value_str.txt', 'w', encoding='utf-8') as f:
        f.write(value_str)
    with open('policy_str.txt', 'w', encoding='utf-8') as f:
        f.write(policy_str)



class PolicyIteration:
    """ Policy iteration """
    def __init__(self, ncol, nrow, P, theta, gamma):
        self.ncol = ncol
        self.nrow = nrow
        self.P = P

        self.v = [0] * self.ncol * self.nrow  # The initialization value is 0
        self.pi = [[0.25, 0.25, 0.25, 0.25]
                   for i in range(self.ncol * self.nrow)]  # Initialize as a uniform random strategy
        self.theta = theta  # Policy evaluation convergence threshold
        self.gamma = gamma  # discount factor

    def policy_evaluation(self):  # Policy evaluation
        cnt = 1  #counter
        while True:
            delta = 0
            # 遍历所有状态
            for s in range(self.ncol * self.nrow):
                v = self.v[s]
                new_v = 0
                # 对于当前策略下的每个动作
                for a in range(4):
                    prob = self.pi[s][a]
                    # 如果该动作的概率为0，跳过
                    if prob == 0:
                        continue
                    # 对于该动作下的所有可能的下一个状态
                    for p, next_state, reward, done in self.P[s][a]:
                        # 计算新的状态价值
                        new_v += prob * p * (reward + self.gamma * (0 if done else self.v[next_state]))
                # 更新最大差值
                delta = max(delta, abs(v - new_v))
                self.v[s] = new_v
            cnt += 1
            # 如果最大差值小于阈值，则停止迭代
            if delta < self.theta:
                break
        print("策略评估进行%d轮后完成" % cnt)

    def policy_improvement(self):  # Policy Improvement
        # 遍历所有状态
        for s in range(self.ncol * self.nrow):
            old_action = copy.deepcopy(self.pi[s])
            # 计算每个动作的价值
            action_values = [0] * 4
            for a in range(4):
                for p, next_state, reward, done in self.P[s][a]:
                    action_values[a] += p * (reward + self.gamma * (0 if done else self.v[next_state]))
            # 找到最优动作
            best_a = max(range(4), key=lambda x: action_values[x])
            # 更新策略为确定性策略
            self.pi[s] = [1 if a == best_a else 0 for a in range(4)]
        print("策略提升完成")
        return self.pi

    def policy_iteration(self):  # Policy iteration
        while 1:
            self.policy_evaluation()
            old_pi = copy.deepcopy(self.pi)  
            new_pi = self.policy_improvement()
            if old_pi == new_pi: break




class ValueIteration:
    """ Value Iteration """
    def __init__(self, ncol, nrow, P, theta, gamma):
        self.ncol = ncol
        self.nrow = nrow
        self.P = P

        self.v = [0] * self.ncol * self.nrow  # The initialization value is 0
        self.theta = theta  # Value convergence threshold
        self.gamma = gamma
        # The Policy obtained after the value iteration is completed
        self.pi = [None for i in range(self.ncol * self.nrow)]

    def value_iteration(self):
        cnt = 0  #counter
        while True:
            cnt += 1
            delta = 0
            # 遍历所有状态
            for s in range(self.ncol * self.nrow):
                old_v = self.v[s]
                # 计算所有动作的价值
                action_values = [0] * 4
                for a in range(4):
                    for p, next_state, reward, done in self.P[s][a]:
                        action_values[a] += p * (reward + self.gamma * (0 if done else self.v[next_state]))
                # 选择最大价值作为新的状态价值
                self.v[s] = max(action_values)
                # 更新最大差值
                delta = max(delta, abs(old_v - self.v[s]))
            # 如果最大差值小于阈值，则停止迭代
            if delta < self.theta:
                break
        print("价值迭代一共进行%d轮" % cnt)
        self.get_policy()


    def get_policy(self):  # Derive a greedy Policy based on the value function
        # 遍历所有状态
        for s in range(self.ncol * self.nrow):
            # 计算每个动作的价值
            action_values = [0] * 4
            for a in range(4):
                for p, next_state, reward, done in self.P[s][a]:
                    action_values[a] += p * (reward + self.gamma * (0 if done else self.v[next_state]))
            # 找到最优动作
            best_a = max(range(4), key=lambda x: action_values[x])
            # 更新策略为确定性策略
            self.pi[s] = [1 if a == best_a else 0 for a in range(4)]





# Don't change code here !
def main():

    # Test your PolicyIteration
    env = CliffWalkingEnv()
    action_meaning = ['^', 'v', '<', '>']
    theta = 0.001 
    gamma = 0.9  
    agent = PolicyIteration(env.ncol, env.nrow, env.P, theta, gamma)
    agent.policy_iteration()
    print_agent(agent, action_meaning, list(range(37, 47)), [47])  
    
    input("Policy Iteration Done\nPress Enter to continue...")

    # Test your ValueIteration
    env = CliffWalkingEnv()
    action_meaning = ['^', 'v', '<', '>']
    theta = 0.001 
    gamma = 0.9  
    agent = ValueIteration(env.ncol, env.nrow, env.P, theta, gamma)
    agent.value_iteration()
    print_agent(agent, action_meaning, list(range(37, 47)), [47]) 


if __name__ == '__main__':
    main()