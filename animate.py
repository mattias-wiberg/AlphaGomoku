import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.animation as animation
import numpy as np

from gameboard import GameBoard

# ! Call after setting labels using ax.set_xticklabels(labels)
def center_labels(ax):
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
def set_square_grid(ax, size):
    major_ticks = np.arange(-0.5, size-0.5, 1)
    labels = np.arange(0, size, 1)
    ax.set_xticks(major_ticks)
    ax.set_xticklabels(labels)

    ax.set_yticks(major_ticks)
    ax.set_yticklabels(labels)
    ax.grid(which='both')

N_row = 15
N_col = 15
object = pd.read_pickle(r'deep_network/moves_tots.p')
#plt.plot(object)
print(len(object))
#plt.waitforbuttonpress()
gameboard = GameBoard(N_row, N_col)
gameboard.board = np.random.randint(-1,2,(N_row,N_col))

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

set_square_grid(ax, N_row)
center_labels(ax)

im = plt.imshow(gameboard.board, cmap='Greys_r')
def init():
    im.set_data(gameboard.board)
    plt.colorbar()

def animate(i):
    gameboard.board = np.random.randint(-1,2,(N_row,N_col))
    im.set_data(gameboard.board)
    return im

#anim = animation.FuncAnimation(fig, animate, init_func=init, frames=1, interval=1)