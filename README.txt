# Requirements
Python 3.10+ (Development in 3.10.9)

> TicTacToeMDP.py
> TicTacToeEnv.py
> RLV2.py (forked from bettermdptools)
> PlannerV2.py (forked from bettermdptools)

# Environment Setup
conda env create -f environment.yml --force
conda activate mdp

# Usage
python TicTacToeMDP.py -type v (or p)
- This program created a TicTacToe game and the MDP problem. It is capable of using either
value iteration or policy iteration to solve the TicTacToe then you play the robot after solving.
- Default is v (value iteration)

python TicTacToeEnv.py
- This file sets up the MDP runs Q-learning on the MDP then sees how well the Q-learning space explores the space by looking at the non-zero moves
and illegal moves.

PlannerV2.py
Forked class of bettermdptools implmentation of policy and value iteration, added prints to be verbose.

RLV2.py
Forked class of bettermdptools implmentation of q-learning to only consider legal Tic-Tac-Toe moves.
