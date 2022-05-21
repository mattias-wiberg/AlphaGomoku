from cv2 import repeat
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

from agent import TDQNAgent
from gameboard import GameBoard

gameboard = GameBoard(15, 15, False, True)
save_path = 'networks/trained/'
network = 'deep_network_conv/'
save_path += network
agent = TDQNAgent(gameboard, save_path=save_path, device="cpu", epsilon_scale=0)
agent.load_strategy(save_path+"qn.pth", 
                    save_path+"moves_tots.p",
                    save_path+"wins.p",
                    save_path+"black_win_frac.p",
                    save_path+"epsilons.p")

nSeconds = 35
fps = 1

def animate(i):
    gameboard.plot()
    agent.turn()
    return [gameboard.im, gameboard.im2]

anim = animation.FuncAnimation(gameboard.fig, animate, frames=nSeconds*fps, interval=1000/fps, repeat=False)
plt.show()
print("Saving animation...")
anim.save('animation.gif', fps=fps, writer='pillow')