from gameboard import GameBoard
from agent import TDQNAgent

visualize = True
strategy_file = 'qn.pth'

gameboard = GameBoard(15, 15)
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
    




