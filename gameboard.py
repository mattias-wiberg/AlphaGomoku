import numpy as np

class GameBoard:
    def __init__(self,N_row,N_col):
        self.N_row=N_row
        self.N_col=N_col
        # Create table for game board
        # -1   black
        #  0   empty
        #  1   white
        self.board = np.zeros((N_row, N_col), dtype=np.int8)    # assignment to tensors not allowed, so must use numpy
        self.gameOver = False
        self.piece = -1

    def restart(self):
        self.board = np.zeros((self.N_row, self.N_col), dtype=np.int8)
        self.gameOver = False
        self.piece = -1

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

    def move(self, row, col):
        assert self.board[row,col] == 0     # double check that it is a legal move
        self.board[row,col] = self.piece
        reward = self.get_reward(row, col)
        if reward != 0:
            self.restart()
        else:
            self.piece *= -1 # Alternate between white and black
        # TODO: reward MUST have a sign for testing cases to work correctly, so take that into account when adding it into the transition
        return reward
