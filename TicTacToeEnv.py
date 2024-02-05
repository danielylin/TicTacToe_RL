import gym
from gym import spaces
import numpy as np
import logging
from TicTacToeMDP import TicTacToeMDP
# from algorithms.rl import RL
from RLV2 import RL

logging.basicConfig(
    format='[%(levelname)s] %(asctime)s: %(message)s', level=logging.DEBUG)


class TicTacToeEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(TicTacToeEnv, self).__init__()

        # Action space: 9 possible actions corresponding to board positions
        self.action_space = spaces.Discrete(9)

        # Observation space: 19683 possible states (3^9)
        self.observation_space = spaces.Discrete(19683)

        self.game = TicTacToeMDP()

    def step(self, action):
        # Apply the action to the board and calculate the next state
        current_board = self.game.board
        i, j = self.game.convert_action_to_index(action)

        # if current_board[i, j] != 0:
        #     end_state = self.game._discretize_board(current_board)
        #     return end_state, -100, True, False, {}

        next_board = np.copy(current_board)
        next_board[i, j] = self.game.computer_move
        next_state = self.game._discretize_board(next_board)

        game_over, winner = self.game._is_game_over(next_board)
        if game_over:
            if winner == 1:
                _, r = self.game.get_reward(next_board)
                self.game.board = next_board
            if winner == 0:
                _, r = self.game.get_reward(next_board)
                self.game.board = next_board
            if winner == 2:
                _, r = self.game.get_reward(next_board)
                self.game.board = next_board
        else:
            open_positions = self.game._get_positions(next_board, 0)
            open_position_list = []
            for pos in open_positions:
                open_position_list.append(
                    self.game.convert_index_to_action(pos[0], pos[1]))

            human_action = np.random.choice(open_position_list)
            human_pos_action = self.game.convert_action_to_index(human_action)
            sim_board = np.copy(next_board)
            sim_board[human_pos_action] = self.game.human_move
            next_state = self.game._discretize_board(sim_board)
            game_over, r = self.game.get_reward(sim_board)

            self.game.board = sim_board

        return next_state, r, game_over, False, {}

    def reset(self):
        self.game.reset_board()
        return self.game._discretize_board(self.game.board), {}

    def render(self, mode="human"):
        self.game.print_board()

    def close(self):
        pass


if __name__ == "__main__":
    tictactoe = TicTacToeEnv()

    # Create an instance of the RL class
    rl_agent = RL(tictactoe)

    for ep in [0, 0.2, 0.5, 1]:
        Q, V, pi, Q_track, pi_track = rl_agent.q_learning(
            init_epsilon=ep,
            min_epsilon=0.01,
            n_episodes=20000)
        error = 0
        num_zeros = 0
        for i in range(0, tictactoe.game.num_states):
            action_p = pi(i)
            idx_p = tictactoe.game.convert_action_to_index(action_p)
            board = tictactoe.game._undiscretize_board(i)
            if action_p == 0:
                num_zeros += 1
            if board[idx_p] != 0:
                error += 1

        print(f"The number of action 0 made for episolon {ep} is {num_zeros}")
        print(f"The number of illegal moves made for episolon {ep} is {error}")

    for ratio in [0.1, 0.25, 0.5, 0.75, 0.9]:
        Q, V, pi, Q_track, pi_track = rl_agent.q_learning(
            init_epsilon=1,
            epsilon_decay_ratio=ratio,
            min_epsilon=0.01,
            n_episodes=20000)
        error = 0
        num_zeros = 0
        for i in range(0, tictactoe.game.num_states):
            action_p = pi(i)
            idx_p = tictactoe.game.convert_action_to_index(action_p)
            board = tictactoe.game._undiscretize_board(i)
            if action_p == 0:
                num_zeros += 1
            if board[idx_p] != 0:
                error += 1

        print(f"The number of action 0 made for episolon decay {ratio} is {num_zeros}")
        print(f"The number of illegal moves made for episolon decay {ratio} is {error}")

