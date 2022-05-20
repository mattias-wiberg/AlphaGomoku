import numpy as np
import matplotlib.pyplot as plt
import matplotlib

class GameBoard:
    def __init__(self,N_row,N_col, interactive=False, draw=False):
        self.N_row=N_row
        self.N_col=N_col
        # Create table for game board
        # -1   black
        #  0   empty
        #  1   white
        self.board = np.zeros((N_row, N_col), dtype=np.int8)    # assignment to tensors not allowed, so must use numpy
        self.gameover = False
        self.piece = -1
        self.n_moves = 0

        # Plotting
        self.fig = plt.figure()
        ax = self.fig.add_subplot(1, 1, 1)

        self.set_square_grid(ax, N_row)
        self.center_labels(ax, self.fig)
        if draw:
            self.im = plt.imshow(self.board, cmap='Greys_r')
            plt.colorbar()
        else:
            self.im = None

        # Playing
        if interactive:
            self.fig.canvas.mpl_connect('button_press_event', self.onclick)

    def restart(self):
        self.board = np.zeros((self.N_row, self.N_col), dtype=np.int8)
        self.gameover = False
        self.piece = -1
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
        padded_board = np.pad(self.board, 4)
        row += 4
        col += 4
        row_bounds = (row-4, row+5)
        col_bounds = (col-4, col+5)
        
        row_victory = self.get_seq_reward( padded_board[row_bounds[0]:row_bounds[1], col] )
        if row_victory != 0:
            return row_victory
        
        col_victory = self.get_seq_reward( padded_board[row, col_bounds[0]:col_bounds[1]] )
        if col_victory != 0:
            return col_victory

        relevant_board = padded_board[row_bounds[0]:row_bounds[1], col_bounds[0]:col_bounds[1]]
        back_diag_victory = self.get_seq_reward( np.diag(relevant_board) )  # \
        if back_diag_victory != 0:
            return back_diag_victory

        forward_diag_victory = self.get_seq_reward( np.diag(np.fliplr(relevant_board)) )    # /
        if forward_diag_victory != 0:
            return forward_diag_victory
        
        return 0

    # ! Call after setting labels using ax.set_xticklabels(labels)
    def center_labels(self, ax, fig):
        # Create offset transform by 10 points in x direction
        dx = 10/72.; dy = 0/72. 
        x_offset = matplotlib.transforms.ScaledTranslation(dx, dy, fig.dpi_scale_trans)
        # Create offset transform by 10 points in y direction
        dx = 0/72.; dy = -10/72. 
        y_offset = matplotlib.transforms.ScaledTranslation(dx, dy, fig.dpi_scale_trans)
        # apply offset transform to all x ticklabels.
        for label in ax.xaxis.get_majorticklabels():
            label.set_transform(label.get_transform() + x_offset)

        # apply offset transform to all x ticklabels.
        for label in ax.yaxis.get_majorticklabels():
            label.set_transform(label.get_transform() + y_offset)

    # Make a square lattice grid at whole integers
    def set_square_grid(self, ax, size):
        major_ticks = np.arange(-0.5, size-0.5, 1)
        labels = np.arange(0, size, 1)
        ax.set_xticks(major_ticks)
        ax.set_xticklabels(labels)

        ax.set_yticks(major_ticks)
        ax.set_yticklabels(labels)
        ax.grid(which='both')

    def onclick(self, event):
        ix, iy = event.xdata, event.ydata
        ix = round(ix)
        iy = round(iy)
        self.move(iy, ix)
        print('x = %f, y = %f, %d'%(ix, iy, self.board[iy, ix]))

    def plot(self):
        self.im.set_data(self.board)

    def move(self, row, col):
        assert self.board[row,col] == 0     # double check that it is a legal move
        
        self.board[row,col] = self.piece
        self.n_moves += 1

        if self.n_moves >= 9:   # need at least 9 moves to win
            reward = self.get_reward(row, col)
        else:
            reward = 0

        if reward == 0 and self.n_moves == self.N_row*self.N_col: # Tie!
            print("Tie!")
            self.gameover = True
            return reward
            
        if reward != 0:
            self.gameover = True
            self.piece *= -1
        else:
            self.piece *= -1 # Alternate between white and black
        return reward
