from gameboard import GameBoard
from agent import TDQNAgent

visualize = False
strategy_file = 'qn.pth'

gameboard = GameBoard(15, 15)
if strategy_file:
    agent = TDQNAgent(gameboard, episode_count=50000, epsilon_scale=1)
else:
    agent = TDQNAgent(gameboard=gameboard)
if strategy_file:
    agent.load_strategy(strategy_file)
    while True:
        agent.turn()
        if visualize:
            gameboard.plot()
else:
    while True:
        agent.turn()
    




