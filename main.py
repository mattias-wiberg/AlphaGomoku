from gameboard import GameBoard
from agent import TDQNAgent
from matplotlib import pyplot as plt

visualize = True
strategy_file = 'qn.pth'

human = True
human_start_piece = 1

gameboard = GameBoard(15, 15, human, visualize)
agent = TDQNAgent(gameboard, device="cpu", epsilon_scale=0)

if strategy_file:
    agent.load_strategy(strategy_file, "moves_tots.p", "wins.p", "black_win_frac.p")
    if human:
        if human_start_piece == 1:
            while True:
                agent.turn()
                gameboard.plot()
                plt.waitforbuttonpress()
                gameboard.plot()

        if human_start_piece == -1:
            while True:
                plt.waitforbuttonpress()
                gameboard.plot()
                agent.turn()
                gameboard.plot()
    
    else:
        while True:
            agent.turn()
            if visualize:
                gameboard.plot()
                plt.waitforbuttonpress()
else:
    while True:
        agent.turn()
    




