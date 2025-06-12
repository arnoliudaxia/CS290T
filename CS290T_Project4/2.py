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
import pdb

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

tictactoe_game = TicTacToeGame(n = 3)


random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

total_matches = 100
total_matches = 10

args = dotdict({'numMCTSSims': 50, 'cpuct': 1.0})
mcts = PureMCTS(tictactoe_game, args)

random_player = TicTacToePlayers.RandomPlayer(tictactoe_game).play
mcts_player = lambda x: np.argmax(mcts.getActionProb(x, temp=0))
arena = Arena(mcts_player, random_player, tictactoe_game, display=TicTacToeGame.display)
# win, lose, tie = arena.playGames(total_matches, verbose=False)
win, lose, tie = arena.playGames(total_matches, verbose=True)

print(f"vs random win: {win}, tie: {tie}, lose: {lose}")
if (win + tie) > total_matches * 0.95:
    print("Implamentation of MCTS is totally correct")
else:
    raise Exception("Implamentation of MCTS might be wrong")