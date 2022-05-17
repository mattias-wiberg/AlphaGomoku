from gameboard import GameBoard
from agent import TDQNAgent

visualize = True
strategy_file = 'qn_mean_13_2.pth'

human = 1
human_start_piece = -1

gameboard = GameBoard(15, 15)

if strategy_file:
    agent = TDQNAgent(gameboard, episode_count=50000, epsilon_scale=0)
else:
    agent = TDQNAgent(gameboard=gameboard)

if strategy_file:
    agent.load_strategy(strategy_file)
    if human:
        if human_start_piece == -1:
            while True:
                agent.turn()
                if visualize:
                    gameboard.plot()
                x, y = [int(x) for x in input().split()]
                agent.turn(forced_move=[x,y])
                if visualize:
                    gameboard.plot()

        if human_start_piece == 1:
            while True:
                x, y = [int(x) for x in input().split()]
                agent.turn(forced_move=[x,y])
                if visualize:
                    gameboard.plot()
                agent.turn()
                if visualize:
                    gameboard.plot()
    
    else:
        while True:
            agent.turn()
            if visualize:
                gameboard.plot()
else:
    while True:
        agent.turn()
    




