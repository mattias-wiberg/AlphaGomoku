from gameboard import GameBoard
import numpy as np
import matplotlib.pyplot as plt
from agent import TDQNAgent
import torch

def visualize_moves(N_row, N_col, moves):
    board = np.zeros((N_row, N_col))
    piece = -1
    for move in moves:
        board[move[0],move[1]] = piece
        piece *= -1
    print(board)
    plt.imshow(board)
    plt.show()

            # black, white
test_moves = [

            [(0,0), (0,1),
            (1,0), (0,2),
            (2,0), (0,3),
            (3,0), (0,4),
            (4,0)],         # vertical test (idx=0)
            
            [(0,0), (1,5),
            (0,2), (2,5),
            (0,3), (3,5),
            (0,4), (4,5),
            (0,5), (5,5)],         # vertical test (idx=1)

            [(6,6), (13,6),
            (7,6), (12,6),
            (8,6), (11,6),
            (9,6), (10,6)],         # vertical test (idx=2)

            [(14,14), (13,6),
            (13,14), (12,6),
            (12,14), (11,6),
            (11,14), (10,6),
            (10,14), (0,0)],         # vertical test (idx=3)

            [(6,6), (1,0),
            (7,6), (3,3),
            (8,6), (5,6),
            (9,6), (10,6),         
            (11,6), (0,6),
            (12,6), (0,7),
            (13,6), (6,9),
            (14,6), (10,10)],   # vertical test (idx=4)

            [(0,14), (1,0),
            (1,13), (3,3),
            (2,12), (5,6),
            (3,11), (10,6),         
            (4,10)],           # / test (idx=5)

            [(0,14), (4,10),
            (1,13), (3,3),
            (2,12), (5,6),
            (3,11), (10,6),         
            (5,9)],          # / test (idx=6)

            [(0,14), (7,4),
            (1,13), (8,5),
            (2,12), (9,6),
            (3,11), (10,7),         
            (5,9), (11,8)],  # \ test (idx=7)

            [(0,14), (10,14),
            (1,13), (9,13),
            (2,12), (8,12),
            (3,11), (7,11),         
            (5,9), (6,10)],  # \ test (idx=8)

            [(14,3), (10,14),
            (14,4), (9,13),
            (14,5), (8,12),
            (14,6), (7,11),         
            (14,7), (0,0)],  # horizontal test (idx=9)

            [(0,0), (6,14),
            (14,4), (6,13),
            (14,5), (6,12),
            (14,6), (6,11),         
            (14,7), (6,10)]  # horizontal test (idx=10)

]

expected_outcomes = [-1, 1, 0, -1, 0, -1, 0, 1, 1, -1, 1]
N_row = 15
N_col = 15

for idx, test in enumerate(test_moves):
    gameboard = GameBoard(N_row, N_col)
    for move in test:
        reward = gameboard.move(move[0], move[1])
        if reward != 0:
            break   
    assert reward == expected_outcomes[idx]
print("Passed gameboard.move()")

#visualize_moves(N_row, N_col, test_moves[10])


seqs_to_test = [
                [1,1,1,1,0,1,1,1,1],            # 0
                [-1,-1,-1,-1,1,-1,-1,-1,-1],    # 1
                [-1,1,1,1,1,0,1,1,1],           # 2
                [0,1,1,1,1,0,1,1,1],            # 3
                [0,1,1,1,1,1,-1,-1,-1],         # 4
                [-1,1,1,1,1,1,-1,-1,-1],        # 5
                [-1,1,-1,-1,-1,-1,-1,1,1],      # 6
                [0,1,1,1,1,0],                  # 7
                [0,-1,-1,-1,-1,-1]              # 8
]
piece_to_test = [-1,-1,-1,-1,1,1,-1,1,-1]
expected_seq_outcomes = [0,0,0,0,1,1,-1,0,-1]
gameboard = GameBoard(N_row, N_col)
for idx, seq in enumerate(seqs_to_test):
    gameboard.piece = piece_to_test[idx]
    assert gameboard.get_seq_reward(seq) == expected_seq_outcomes[idx]
print("Passed gameboard.get_seq_reward()")

# Test get random action
gameboard = GameBoard(N_row, N_col)
agent = TDQNAgent(gameboard=gameboard)
for _ in range(10000):
    gameboard.board = np.random.randint(-1,2,(N_row,N_col))
    index = agent.get_random_action()
    assert gameboard.board[index] == 0
print("Passed agent.get_random_action()")

# Test get random action
gameboard = GameBoard(N_row, N_col)
agent = TDQNAgent(gameboard=gameboard)
gameboard.board = np.random.randint(-1,2,(N_row,N_col))
gameboard.plot()

gameboard = GameBoard()
torch.manual_seed(1234)
boards = [(np.random.randint(-1,2,(N_row,N_col)), (0,0)),
            (np.random.randint(-1,2,(N_row,N_col)), (0,0)),
            (np.random.randint(-1,2,(N_row,N_col)), (0,0)),
            (np.random.randint(-1,2,(N_row,N_col)), (0,0)),
            (np.random.randint(-1,2,(N_row,N_col)), (0,0))
]
agent = TDQNAgent(gameboard=gameboard)
for board, target in boards:
    gameboard.board = board
    assert agent.get_max_action() == target
