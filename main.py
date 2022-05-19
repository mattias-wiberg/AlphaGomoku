from gameboard import GameBoard
from agent import TDQNAgent
from matplotlib import pyplot as plt

visualize = False
load_strategy = True
save_path = 'networks/trained/'
network = 'test_network/'
save_path += network

human = False
human_start_piece = 1

gameboard = GameBoard(15, 15, human, visualize)
agent = TDQNAgent(gameboard, save_path=save_path, device="cpu")

if load_strategy:
    agent.load_strategy(save_path+"qn.pth", 
                        save_path+"moves_tots.p",
                        save_path+"wins.p",
                        save_path+"black_win_frac.p",
                        save_path+"epsilons.p")
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
    




