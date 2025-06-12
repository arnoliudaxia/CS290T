import random

import numpy as np
import torch

from alpha_zero.MCTS import MCTS, PureMCTS
from alpha_zero.Coach import Coach
from alpha_zero.Arena import Arena
from alpha_zero.othello.OthelloGame import OthelloGame
from alpha_zero.tictactoe.TicTacToeGame import TicTacToeGame
from alpha_zero.othello.pytorch.NNet import NNetWrapper as nn
from alpha_zero.utils import dotdict

from alpha_zero.tictactoe import TicTacToePlayers
from alpha_zero.othello.OthelloPlayers import GreedyOthelloPlayer

# for auto-reloading external modules
# see http://stackoverflow.com/questions/1907993/autoreload-of-modules-in-ipython
from pathlib import Path

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)


total_matches = 40

args = dotdict({'numMCTSSims': 50, 'cpuct': 1.0})
othello_game = OthelloGame(n = 6)
nnet = nn(othello_game)

nnet.load_checkpoint("othello_6", "best.pth.tar")
# nnet.load_checkpoint("othello_6", "checkpoint_1.pth.tar")
alpha_zero = MCTS(othello_game, nnet, args)
mcts = PureMCTS(othello_game, args)

greed_player = GreedyOthelloPlayer(othello_game).play
mcts_player = lambda x: np.argmax(mcts.getActionProb(x, temp=0))
your_alpha_zero_player = lambda x: np.argmax(alpha_zero.getActionProb(x, temp=0))

arena = Arena(your_alpha_zero_player, greed_player, othello_game, display=OthelloGame.display)

win, lose, tie = arena.playGames(total_matches, verbose=False)
print(f"vs greed win: {win}, tie: {tie}, lose: {lose}")
if win <= total_matches * 0.8:
    raise Exception("Implamentation of alphaZero might be wrong or the hyperparameters are not good enough")

arena = Arena(your_alpha_zero_player, mcts_player, othello_game, display=OthelloGame.display)

win, lose, tie = arena.playGames(total_matches, verbose=False)
print(f"vs mcts win: {win}, tie: {tie}, lose: {lose}")
if win >= total_matches * 0.4:
    print("Implamentation of MCTS is totally correct")
else:
    raise Exception("Implamentation of alphaZero might be wrong or the hyperparameters are not good enough")
