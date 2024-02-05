import numpy as np
import logging
# from algorithms.planner import Planner
from PlannerV2 import Planner
logging.basicConfig(
    format='[%(levelname)s] %(asctime)s: %(message)s', level=logging.DEBUG)
import argparse

class TicTacToeMDP():
    """This class is an MDP representation of Tic-Tac-Toe.
    """
    def __init__(self):
        self.winning_positions = [
            [(0, 0), (0, 1), (0, 2)],  # Rows
            [(1, 0), (1, 1), (1, 2)],
            [(2, 0), (2, 1), (2, 2)],
            [(0, 0), (1, 0), (2, 0)],  # Columns
            [(0, 1), (1, 1), (2, 1)],
            [(0, 2), (1, 2), (2, 2)],
            [(0, 0), (1, 1), (2, 2)],  # Diagonals
            [(0, 2), (1, 1), (2, 0)]
        ]
        self.game_over = False
        self.winning_player = 0  # robot is always player 1, human is player 2
        self.num_states = 3**9
        self.num_actions = 9
        self.P = {
            s: {
                a: [] for a in range(self.num_actions)
            } for s in range(self.num_states)
        }

        self.computer_move = 1
        self.human_move = 2

        self.reset_board()

    def reset_board(self):
        self.board = np.zeros((3, 3))
        self.player_move = 1
        self.game_over = False
        self.winning_player = 0

    def _discretize_board(self, board):
        """
        This method returns the discrete board state.

        Returns:
            board_state_index: int between 0 and 19682 describing the state
            of the board.
        """
        board_state_index = 0
        for i in range(board.shape[0]):
            for j in range(board.shape[1]):
                val = board[i][j]*(3**(3*i+j))
                board_state_index += val
        return int(board_state_index)

    def _is_illegal_state(self, board):
        computer_moves = np.count_nonzero(board == 1)
        human_moves = np.count_nonzero(board == 2)
        result = human_moves - computer_moves

        if result != 0 and result != 1:
            return True
        else:
            return False

    def _undiscretize_board(self, board_state_index):
        """
        This method returns the 3x3 board array for a given state index.

        Returns:
            board: numpy array representing the state of the board
        """
        board = np.zeros((3, 3))
        for i in range(2, -1, -1):
            for j in range(2, -1, -1):
                factor = 3**(3*i+j)
                quotient, board_state_index = divmod(board_state_index, factor)
                board[i][j] = quotient
        return board

        board = np.zeros((3, 3))
        for i in range(2, -1, -1):
            for j in range(2, -1, -1):
                factor = 3**(3*i+j)
                quotient, board_state_index = divmod(board_state_index, factor)
                board[i][j] = quotient
        return board

    def _get_positions(self, board, value=0):
        """
        This method returns a list of positions for a given value.

        Returns:
            board_state_index: list of tuples representing open positions
        """
        open_pos = np.where(board == value)
        i_vals = open_pos[0]
        j_vals = open_pos[1]

        if i_vals.size == 0:
            return []

        pos_list = []

        for idx in range(len(i_vals)):
            pos = (i_vals[idx], j_vals[idx])
            pos_list.append(pos)

        return pos_list

    def _is_game_over(self, board):
        """
        This method checks if the game is over.

        Returns:
            bool, player: winner and winning player (0, 1, 2)
        """
        open_positions = self._get_positions(board, value=0)
        robot_positions = self._get_positions(board, value=1)
        human_positions = self._get_positions(board, value=2)

        if robot_positions is None or human_positions is None:
            return False, 0

        for winning_position in self.winning_positions:
            if set(winning_position).issubset(set(robot_positions)):
                return True, 1
            if set(winning_position).issubset(set(human_positions)):
                return True, 2

        if open_positions is None or len(open_positions) == 0:
            return True, 0
        else:
            return False, 0

    def convert_action_to_index(self, a):
        i, j = a // 3, a % 3
        return i, j

    def convert_index_to_action(self, i, j):
        a = i * 3 + j
        return a

    def does_opp_win(self, board):
        """
        Simulate an opponent move.
        """
        open_positions = self._get_positions(board, value=0)
        for pos in open_positions:
            board_sim = np.copy(board)
            board_sim[pos] = self.human_move
            p2_positions = self._get_positions(board_sim, value=2)
            for winning_position in self.winning_positions:
                if set(winning_position).issubset(set(p2_positions)):
                    return True, board_sim
        return False, None

    def get_reward(self, board, r_win=5, r_lose=-5, r_draw=1, r_not_over=2):
        open_positions = self._get_positions(board, 0)
        computer_positions = self._get_positions(board, 1)
        human_positions = self._get_positions(board, 2)

        for winning_position in self.winning_positions:
            if set(winning_position).issubset(set(computer_positions)):
                return True, r_win
            if set(winning_position).issubset(set(human_positions)):
                return True, r_lose

        if len(open_positions) == 0 or open_positions is None:
            return True, r_draw

        return False, r_not_over

    def print_winning_positions(self):
        for winning_position in self.winning_positions:
            winning_board = np.zeros((3, 3))
            for pos in winning_position:
                winning_board[pos] = 1
            self.print_board(board=winning_board)

    def update_transition_matrix(self, r_win=5, r_draw=1, r_lose=-5,
                                 r_not_over=2, verbose=False):
        """
        Sets the transition matrix.
        """
        illegal_states = 0

        for s in range(self.num_states):
            board = self._undiscretize_board(s)
            open_positions = self._get_positions(board, 0)

            # Check if there are any valid positions
            if open_positions is None or open_positions == 0:
                continue

            if self._is_illegal_state(board):
                illegal_states += 1
                continue

            for a in range(self.num_actions):
                i, j = self.convert_action_to_index(a)
                if (i, j) not in open_positions:
                    continue

                next_board = np.copy(board)
                next_board[i, j] = self.computer_move

                game_over, winner = self._is_game_over(next_board)
                next_state = self._discretize_board(next_board)
                if game_over:
                    if winner == 1:
                        self.P[s][a].append((1, next_state, r_win, True))
                    elif winner == 0:
                        self.P[s][a].append((1, next_state, r_draw, True))
                else:
                    open_positions = self._get_positions(next_board, 0)
                    open_position_list = []
                    for pos in open_positions:
                        open_position_list.append(
                            self.convert_index_to_action(pos[0], pos[1]))

                    p = 1/len(open_position_list)

                    for pos in open_position_list:
                        pos_idx = self.convert_action_to_index(pos)
                        next_board_opp = np.copy(next_board)
                        next_board_opp[pos_idx] = self.human_move
                        opp_next_state = self._discretize_board(next_board_opp)
                        game_over, winner = self._is_game_over(next_board_opp)

                        if game_over:
                            if winner == 1:
                                self.P[s][a].append(
                                    (p, opp_next_state, r_win, True))
                            elif winner == 0:
                                self.P[s][a].append(
                                    (p, opp_next_state, r_draw, True))
                        else:
                            self.P[s][a].append(
                                    (p, opp_next_state, r_not_over, False))

                    # opp_win, opp_win_board = self.does_opp_win(next_board)

                    # if opp_win:
                    #     opp_win_state = self._discretize_board(opp_win_board)
                    #     self.P[s][a].append((1, opp_win_state, r_lose, True))
                    # else:
                    #     next_opp_moves = self._get_positions(next_board, 0)

                    #     p = 1/len(next_opp_moves)
                    #     for pos in next_opp_moves:
                    #         next_board_opp = np.copy(next_board)
                    #         next_board_opp[pos] = self.human_move
                    #         opp_next_state = self._discretize_board(
                    #             next_board_opp)
                    #         self.P[s][a].append(
                    #             (p, opp_next_state, r_not_over, False))

        if verbose:
            logging.info(f"There are {illegal_states} illegal states out of "
                         f"{self.num_states} total states.")

    def user_action(self, player, i, j):
        """This method makes a move for the player."""
        self.board[i, j] = player
        self.discretized_board = self._discretize_board(self.board)

    def computer_action(self, player, policy=None):
        current_state = self._discretize_board(self.board)
        if policy is None:
            open_positions = self._get_positions(self.board, 0)
            open_position_list = []
            for pos in open_positions:
                open_position_list.append(
                    self.convert_index_to_action(pos[0], pos[1]))
            computer_action = np.random.choice(open_position_list)

        else:
            computer_action = policy(current_state)

        i, j = self.convert_action_to_index(computer_action)

        if self.board[i, j] != 0:
            print(
                f"The policy is about to make an illegal move at ({i}, {j})!")
            open_positions = self._get_positions(self.board, 0)
            open_position_list = []
            for pos in open_positions:
                open_position_list.append(
                    self.convert_index_to_action(pos[0], pos[1]))
            new_computer_action = np.random.choice(open_position_list)
            i, j = self.convert_action_to_index(new_computer_action)

        self.board[i, j] = player
        self.discretized_board = self._discretize_board(self.board)

    def print_board(self, player_one_is_x=True, board=None):

        if board is None:
            board = self.board

        symbol_map = {
            0: ' ',
            1: 'X' if player_one_is_x else 'O',
            2: 'O' if player_one_is_x else 'X'
        }

        for row in range(3):
            row_str = ''
            for col in range(3):
                row_str += symbol_map[board[row, col]]
                if col != 2:
                    row_str += '|'
            print(row_str)
            if row != 2:
                print('-'*5)
        print('\n'*1)

    def get_input(self, prompt, options):
        while True:
            response = input(prompt).strip().upper()
            if response in options:
                return response
            else:
                print("Invalid move type! Try again.")

    def play_robot(self, policy=None):
        self.reset_board()
        print("Let's play Tic-Tac-Toe!")
        print("Human plays as: O.")
        robot_first = self.get_input(
            "Robot goes first (T or F): ", {"T", "F"}) == "T"
        if robot_first:
            print("##### ROBOT TURN #####")
            self.computer_action(player=1, policy=policy)
            self.print_board()

        while True:
            print("Your turn!")
            row = int(self.get_input(
                "Enter row (1-3): ", {"1", "2", "3"})) - 1
            col = int(self.get_input(
                "Enter column (1-3): ", {"1", "2", "3"})) - 1
            if self.board[row, col] != 0:
                print("That space is already taken, please choose another.")
                self.print_board()
                continue
            print("##### PLAYER TURN #####")
            self.user_action(2, i=row, j=col)
            self.print_board()
            logging.debug(f"State: {self.discretized_board}")

            game_over, winning_player = self._is_game_over(self.board)
            if game_over:
                if winning_player == 1:
                    print("Ha! You lose!")
                    break
                elif winning_player == 2:
                    print("You win!")
                    break
                else:
                    print("DRAW!")
                    break

            print("##### ROBOT TURN #####")
            self.computer_action(player=1, policy=policy)
            self.print_board()
            logging.debug(f"State: {self.discretized_board}")
            game_over, winning_player = self._is_game_over(self.board)
            if game_over:
                if winning_player == 1:
                    print("Ha! You lose!")
                    break
                elif winning_player == 2:
                    print("You win!")
                    break
                else:
                    print("Draw!")
                    break

def test_policies(discount_fct=0.99):
    game_val = TicTacToeMDP()
    game_val.update_transition_matrix(verbose=True)
    V_val, V_track_val, pi_val = Planner(game_val.P).value_iteration(
        gamma=discount_fct)

    game_pol = TicTacToeMDP()
    game_pol.update_transition_matrix(verbose=True)
    V_pol, V_track_pol, pi_pol = Planner(game_pol.P).policy_iteration(
        gamma=discount_fct
    )

    matches = 0
    non_zero_entry = 0
    for i in range(game_val.num_states):
        if pi_pol(i) == pi_val(i):
            matches += 1
        if pi_pol(i) > 0:
            non_zero_entry += 1

    print(f"The number of matches {matches}")
    print(f"The number of non zero moves {non_zero_entry}")

    return pi_val, V_val, pi_pol, V_pol


def main():
    parser = argparse.ArgumentParser(description='Tic Tac Toe MDP')
    parser.add_argument('-type', choices=['v', 'p'], default='v', help='Type of algorithm (v or p)')

    # Parse the command-line arguments
    args = parser.parse_args()

    # Check the value of the 'type' argument and run the appropriate code

    game = TicTacToeMDP()
    game.update_transition_matrix(
            r_win=5, r_draw=1, r_lose=-5, r_not_over=-0.1, verbose=True)
    if args.type == 'v':
        V_val, V_track_val, pi_val = Planner(game.P).value_iteration(
            gamma=0.99)
    elif args.type == 'p':
        V_val, V_track_val, pi_val = Planner(game.P).policy_iteration(
            gamma=0.99)

    game.play_robot(policy=pi_val)


if __name__ == "__main__":
    main()
