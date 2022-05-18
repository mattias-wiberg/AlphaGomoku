from gameboard import GameBoard
from agent import TDQNAgent

visualize = False
strategy_file = ''

human = 0
human_start_piece = 1

gameboard = GameBoard(15, 15)
agent = TDQNAgent(gameboard, device="cpu", episode_count=50000, epsilon_scale=5000)

if strategy_file:
    agent.load_strategy(strategy_file, "moves_tots.p", "wins.p", "black_win_frac.p")
    if human:
        if human_start_piece == 1:
            while True:
                agent.turn()
                if visualize:
                    gameboard.plot()
                x, y = [int(x) for x in input().split()]
                agent.turn(forced_move=[x,y])
                if visualize:
                    gameboard.plot()

        if human_start_piece == -1:
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
    




