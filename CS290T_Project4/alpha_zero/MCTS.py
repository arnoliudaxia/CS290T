import logging
import math
from typing import Tuple, List

from tqdm import tqdm
import numpy as np
import pdb


from .Game import Game

EPS = 1e-8

log = logging.getLogger(__name__)

class PureMCTS():
    def __init__(self, game: Game, args):
        self.game = game
        self.args = args
        self.Qsa = {}  # stores Q values for s,a (as defined in the paper)
        self.Nsa = {}  # stores #times edge s,a was visited
        self.Ns = {}  # stores #times board s was visited

        self.Ps = {}  # For pure MCTS, Ps is the same with Vs

        self.Es = {}  # stores game.getGameEnded ended for board s
        self.Vs = {}  # stores game.getValidMoves for board s
    
    def load_tree(self, file):
        try:
            self.Qsa, self.Nsa,  self.Ns, self.Ps, self.Es, self.Vs = np.load(file, allow_pickle=True)
        except Exception as e:
            log.error("Error loading tree: %s", e)
            return
    
    def dump_tree(self, file):
        np.save(file, (self.Qsa, self.Nsa, self.Ns, self.Ps, self.Es, self.Vs))
        
    def getActionProb(self, canonicalBoard, temp=1):
        """
        Calculate the probability of each action based on the number of times each action has been selected.

        Args:
            canonicalBoard (np.ndarray): The current state of the game board.
            temp (float, optional): The temperature parameter for the softmax function. A lower temperature results in more deterministic result, 0 means no sample.

        Returns:
            List: A list of probabilities for each action.
        """
        # doing rollout for numMCTSSims times
        for i in range(self.args.numMCTSSims):
            self.search(canonicalBoard)

        s = self.game.stringRepresentation(canonicalBoard)
        counts = [self.Nsa[(s, a)] if (s, a) in self.Nsa else 0 for a in range(self.game.getActionSize())]

        if temp == 0:
            # return the probs with only one of the best actions (break ties randomly)
            bestAs = np.array(np.argwhere(counts == np.max(counts))).flatten()
            bestA = np.random.choice(bestAs)
            probs = [0] * len(counts)
            probs[bestA] = 1
            return probs

        # return the probs with temperature
        # the policy vector where the probability of the ith action is proportional to Nsa[(s,a)]**(1/temp)
        counts = [x ** (1. / temp) for x in counts]
        total = sum(counts)
        probs = [x / total if total > 0 else 0 for x in counts]
        return probs
    
    def select(self, canonicalBoard) -> Tuple[List[Tuple[np.ndarray, int]], np.ndarray]:
        """
        Selects an unexpanded node on the given canonical board.

        Args:
            canonicalBoard (np.ndarray): The current canonical board state.

        Returns:
            Tuple[List[Tuple[np.ndarray, int]], np.ndarray]: A tuple containing the path of moves made and the final board state.
        """
        # The `select`` here totally covers the phase of selection and expansion
        # for ease of implementation
        path = []
        s = self.game.stringRepresentation(canonicalBoard)
        player = 1
        while True:
            # 判断当前节点是否为未扩展节点或终止状态
            # if self.game.getGameEnded(canonicalBoard, 1) != 0:
            # pdb.set_trace()
            if s not in self.Ns:
                self.Ns[s] = 0
                return path, canonicalBoard
            if  self.game.getGameEnded(canonicalBoard, 1) != 0:
                return path, canonicalBoard
            
            # select node by ucb selection to generate a path
            # NOTICE: for the board is always a canonicalBoard, so the current player is always 1
            # use self.game.getNextState(canonicalBoard, 1, action)
            if s in self.Vs:
                valids = self.Vs[s]
            else:
                valids = self.game.getValidMoves(canonicalBoard, 1)
                self.Vs[s] = valids
            
            # 使用UCB公式选择最佳动作
            action = self.ucb_select(s, valids)
            path.append((canonicalBoard, action))
            
            # 获取执行该动作后的下一个状态
            next_board, player = self.game.getNextState(canonicalBoard, 1, action)
            # 转换为标准形式（确保当前玩家始终为1）
            canonicalBoard = self.game.getCanonicalForm(next_board, player)
            
            # 将(状态,动作)对添加到路径中
            # path.append((canonicalBoard, action))
            # 更新当前状态的字符串表示
            s = self.game.stringRepresentation(canonicalBoard)
    
    def expand(self, canonicalBoard, s):
        """
        Expand the search tree by adding the valid moves for the current state.

        Args:
            canonicalBoard (numpy.ndarray): The current state of the board.
            s (str): The state identifier.

        Returns:
            None
        """
        if s not in self.Ns:
            valids = self.game.getValidMoves(canonicalBoard, 1)
            if len(valids) == 0:
                return
            self.Vs[s] = valids
            self.Ns[s] = 0
            

    def simulate(self, canonicalBoard, s):
        """
        Simulate the game from the given state until the terminal state is reached.

        Args:
            canonicalBoard (numpy.ndarray): The current state of the game board.
            s (str): The string representation of the current state.

        Returns:
            int: The reward obtained from the simulation.
        """
        invert_reward = True
        # canonicalBoard = self.game.getCanonicalForm(canonicalBoard, -1)

        while True:
            if s not in self.Es:
                reward = self.game.getGameEnded(canonicalBoard, 1)
                self.Es[s] = reward

            # if the game has ended
            # return the reward
            # NOTICE: beware which one player the reward is for, which can be determined by invert_reward
            if self.Es[s] != 0:

                return self.Es[s] * (1 if not invert_reward else -1)

            if s in self.Vs:
                valids = self.Vs[s]
            else:
                valids = self.game.getValidMoves(canonicalBoard, 1)
                self.Vs[s] = valids

            # get random action from valid moves
            action = np.random.choice(self.game.getActionSize(), p=valids/ np.sum(valids))
            # action = self.ucb_select(s, valids)
            
            # get the next board and update 's'
            # NOTICE: for the board is always a canonicalBoard, so the current player is always 1
            # use self.game.getNextState(canonicalBoard, 1, action)
            next_board, player= self.game.getNextState(canonicalBoard, 1, action)
            canonicalBoard = self.game.getCanonicalForm(next_board, player)
            s = self.game.stringRepresentation(canonicalBoard)
            
            invert_reward = not invert_reward

    def backup(self, path, reward):
        """
        Perform the backup operation for the given path and reward.

        Args:
            path (list): A list of tuples, where each tuple contains a canonical board state and the corresponding action taken.
            reward (float): The reward obtained after taking the action.

        """

        # This method iterates over the path in reverse order, updating Ns, Nsa, and Qsa.
        # NOTICE: the reward is different for different player, so we need to invert it every time
        

        # 然后处理路径上的所有节点
        for board, action in reversed(path):
            board = self.game.getCanonicalForm(board, 1)
            s = self.game.stringRepresentation(board)
                
            if s in self.Ns:
                self.Ns[s] += 1
            else:
                self.Ns[s] = 0
                
            # s=self.game.stringRepresentation(nowState)

            if (s, action) in self.Qsa:
                # self.Qsa[(s, action)] = (self.Qsa[(s, action)] * (self.Nsa[(s, action)]-2) + reward) / (1+self.Nsa[(s, action)])
                # self.Qsa[(s, action)] = (self.Nsa[(s, action)]*(self.Qsa[(s, action)]-reward)  ) / (self.Nsa[(s, action)]+1)
                # self.Qsa[(s, action)] =(reward ) / (self.Nsa[(s, action)])
                self.Qsa[(s, action)] =(self.Nsa[(s, action)]*(self.Qsa[(s, action)]+reward)  ) / (self.Nsa[(s, action)])
                self.Nsa[(s, action)] += 1
                
            else:
                self.Qsa[(s, action)] = reward
                self.Nsa[(s, action)] = 1
                
            reward = -reward  # 为对手玩家反转奖励
            
            

    def search(self, canonicalBoard):
        """
        Perform a search on the given canonical board.
        Doing select, expand, simulate and backup in sequence.

        Args:
            canonicalBoard (object): The current state of the game board.

        Returns:
            None: This method does not return a value.
        """
        # do selection, expansion and simulation
        path, leaf_board = self.select(canonicalBoard)
        # pdb.set_trace()
        s = self.game.stringRepresentation(leaf_board)
        reward = self.simulate(leaf_board, s)

        # expand the node for the root
        if len(path) == 0:
            return

        # do backup
        self.backup(path, reward)
        
    def ucb_select(self, s: str, validMoves: np.ndarray) -> int:
        """
        Selects the action with the highest Upper Confidence Bound (UCB) for a given state.

        Args:
            s (str): The string representation of the current state.
            validMoves (np.ndarray): A binary array where each index represents whether 
                                     the corresponding action is valid (1) or not (0).

        Returns:
            int: The index of the action with the highest UCB.
        """
        cur_best = -float('inf')
        best_act = -1
        
        # score = Qsa + cpuct * sqrt(Ns / Nsa)
        # NOTICE: we always select the `first` action that has not been visited
        valid_actions = np.where(validMoves == 1)[0]
        
        for a in valid_actions:
            if (s, a) not in self.Nsa:
                return a
            
            # Calculate UCB score
            if (s,a) in self.Qsa:
                ucb_score = self.Qsa[(s, a)] + self.args.cpuct * math.sqrt((self.Ns[s]) / self.Nsa[(s, a)])
            else:
                ucb_score = self.Qsa[(s, a)]
                
            if ucb_score > cur_best:
                cur_best = ucb_score
                best_act = a
        
                
        return best_act


class MCTS(PureMCTS):
    """
    This class handles the MCTS tree for AlphaZero.
    """

    def __init__(self, game, nnet, args):
        self.game = game
        self.nnet = nnet
        self.args = args
        self.Qsa = {}  # stores Q values for s,a (as defined in the paper)
        self.Nsa = {}  # stores #times edge s,a was visited
        self.Ns = {}  # stores #times board s was visited
        self.Ps = {}  # stores initial policy (returned by neural net)

        self.Es = {}  # stores game.getGameEnded ended for board s
        self.Vs = {}  # stores game.getValidMoves for board s


    def simulate(self, canonicalBoard, s):
        """
        Simulate the game from the given state until the terminal state is reached.

        Args:
            canonicalBoard (numpy.ndarray): The current state of the game board.
            s (str): The string representation of the current state.

        Returns:
            int: The reward obtained from the simulation.
        """
        
        # doing simulation like pure mcts
        # NOTICE: use nnet to get policy and reward
        # NOTICE: store the policy into Ps
        # NOTICE: there is no need to simulate unitl the game ends
        if s not in self.Es:
            reward = self.game.getGameEnded(canonicalBoard, 1)
            self.Es[s] = reward
        
        if self.Es[s] != 0:
            return self.Es[s]
        
        if s not in self.Ps:
            # Get policy and value from neural network
            policy, value = self.nnet.predict(canonicalBoard)
            
            # 确保valid moves已经初始化
            if s not in self.Vs:
                valids = self.game.getValidMoves(canonicalBoard, 1)
                self.Vs[s] = valids
            else:
                valids = self.Vs[s]
            
            # Normalize policy and store it
            policy = policy * valids  # Mask invalid moves
            sum_policy = np.sum(policy)
            if sum_policy > 0:
                policy /= sum_policy  # Renormalize
            else:
                # If all valid moves were masked, use uniform probability for valid moves
                policy = valids / np.sum(valids)
            
            self.Ps[s] = policy
            
            return -value  # Return negative value because it's from opponent's perspective
        
        # 如果状态已经在Ps中，说明已经被神经网络评估过，直接返回0
        # 实际上这种情况在正常的MCTS流程中不应该发生
        return 0

    def ucb_select(self, s: str, validMoves: np.ndarray) -> int:
        # ucb select formula: u = value + cpuct * P * sqrt(N) / (1 + Nsa)
        cur_best = -float('inf')
        best_act = -1
        valid_actions = np.where(validMoves == 1)[0]
        
        for a in valid_actions:
            if (s, a) not in self.Nsa:
                # 对于未访问过的动作，给予最高优先级
                ucb_score = float('inf')
                p_sa = self.Ps[s][a]
            else:
                q_sa = self.Qsa[(s, a)]
                p_sa = self.Ps[s][a]
                n_sa = self.Nsa[(s, a)]
                n_s = self.Ns[s]
                
                # # 确保策略已经计算过
                # if s in self.Ps:
                #     p_sa = self.Ps[s][a]
                # else:
                #     # 如果策略未计算，给予均匀概率
                #     p_sa = 1.0 / len(valid_actions)
                
                ucb_score = q_sa + self.args.cpuct * p_sa * math.sqrt(n_s) / (1 + n_sa)
            
            if ucb_score > cur_best:
                cur_best = ucb_score
                best_act = a
        
        return best_act

