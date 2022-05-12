from gameboard import GameBoard
from agent import TDQNAgent

gameboard = GameBoard(15, 15)
agent = TDQNAgent(gameboard=gameboard)
while True:
    agent.turn()

