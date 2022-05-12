import numpy as np
import imagesc as sc
import copy

class GameBoard:
    def __init__(self,N_row,N_col):
        self.N_row=N_row
        self.N_col=N_col
        # Create table for game board
        # -1   black
        #  0   empty
        #  1   white
        self.board = np.zeros((N_row, N_col), dtype=np.int8)    # assignment to tensors not allowed, so must use numpy
        self.gameover = False
        self.piece = -1
        self.black_to_play_history = []
        self.black_move_history = []
        self.white_to_play_history = []
        self.white_move_history = []
        self.n_moves = 0

    def restart(self):
        self.board = np.zeros((self.N_row, self.N_col), dtype=np.int8)
        self.gameover = False
        self.piece = -1
        self.black_to_play_history = []
        self.black_move_history = []
        self.white_to_play_history = []
        self.white_move_history = []
        self.n_moves = 0

    def get_seq_reward(self, sequence):
        consecutive_seq_sum = np.sum(sequence[0:5])
        if abs(consecutive_seq_sum) == 5:
            return consecutive_seq_sum/5
        for i in range(5, len(sequence)):
            next_val = sequence[i]
            if next_val != self.piece:
                # next grid is either empty or occupied by the opposing player, victory not possible
                return 0
            else:
                consecutive_seq_sum -= sequence[i-5]    # subtract the first value from the rolling window 
                consecutive_seq_sum += next_val         # add the new value to the rolling window
                if abs(consecutive_seq_sum) == 5:
                    return consecutive_seq_sum/5
        return 0 # no one has won

    # -1/0/+1: black wins/nothing/white wins
    def get_reward(self, row, col):
        row_bounds = (max(0,row-4), min(self.N_row,row+5))
        col_bounds = (max(0,col-4), min(self.N_col,col+5))
        
        row_victory = self.get_seq_reward( self.board[row_bounds[0]:row_bounds[1], col] )
        if row_victory != 0:
            return row_victory
        
        col_victory = self.get_seq_reward( self.board[row, col_bounds[0]:col_bounds[1]] )
        if col_victory != 0:
            return col_victory

        relevant_board = self.board[row_bounds[0]:row_bounds[1], col_bounds[0]:col_bounds[1]]
        back_diag_victory = self.get_seq_reward( np.diag(relevant_board) )  # \
        if back_diag_victory != 0:
            return back_diag_victory

        forward_diag_victory = self.get_seq_reward( np.diag(np.fliplr(relevant_board)) )    # /
        if forward_diag_victory != 0:
            return forward_diag_victory
        
        return 0

    def plot(self):
        sc.plot(
            self.board,
            grid=True,
            linewidth=1,
            title="Turn: {0}".format(self.piece),
            cmap="Greys",
            figsize=(5, 5),
        )

    def move(self, row, col):
        # TODO: returns 0 on a draw, is that OK? think it should be OK!
        assert self.board[row,col] == 0     # double check that it is a legal move
        
        if self.piece == -1:
            self.black_to_play_history.append(copy.deepcopy(self.board))
            self.black_move_history.append((row, col))
            if len(self.black_to_play_history) == 3:
                self.black_to_play_history.pop(0)
                self.black_move_history.pop(0)
        elif self.piece == 1:
            self.white_to_play_history.append(copy.deepcopy(self.board))
            self.white_move_history.append((row, col))
            if len(self.white_to_play_history) == 3:
                self.white_to_play_history.pop(0)
                self.white_move_history.pop(0)

        self.board[row,col] = self.piece
        self.n_moves += 1
        if self.n_moves >= 9:
            reward = self.get_reward(row, col)
        else:
            reward = 0
        if reward != 0:
            self.gameover = True
            self.piece *= -1
        else:
            self.piece *= -1 # Alternate between white and black
        return reward
