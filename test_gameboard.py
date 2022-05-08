from gameboard import GameBoard
import numpy as np
import matplotlib.pyplot as plt

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


]

expected_outcomes = [-5, 5, 0, -5, 0]

N_row = 15
N_col = 15

for idx, test in enumerate(test_moves):
    gameboard = GameBoard(N_row, N_col)
    for move in test:
        reward = gameboard.move(move[0], move[1])
        if reward != 0:
            break   
    assert reward == expected_outcomes[idx]
    print(f"Passed {idx+1}/{len(test_moves)}")

#visualize_moves(N_row, N_col, test_moves[4])

