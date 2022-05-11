from gameboard import GameBoard
import numpy as np
import matplotlib.pyplot as plt
from agent import TDQNAgent
import time
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

N_row = 15
N_col = 15

def test_move():
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

    for idx, test in enumerate(test_moves):
        gameboard = GameBoard(N_row, N_col)
        for move in test:
            reward = gameboard.move(move[0], move[1])
            if reward != 0:
                break   
        assert reward == expected_outcomes[idx]
    print("Passed gameboard.move()")

    #visualize_moves(N_row, N_col, test_moves[10])

def test_get_seq_reward():
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
def test_get_random_action():
    gameboard = GameBoard(N_row, N_col)
    agent = TDQNAgent(gameboard=gameboard)
    time_total = 0
    for _ in range(10000):
        gameboard.board = np.random.randint(-1,2,(N_row,N_col))
        start_time = time.time()
        index = agent.get_random_action()
        time_total += time.time() - start_time
        assert gameboard.board[index] == 0
    print("Passed agent.get_random_action()")
    print("Total time: ", time_total)

# Test plot_board
def test_plot_board():
    gameboard = GameBoard(N_row, N_col)
    agent = TDQNAgent(gameboard=gameboard)
    gameboard.board = np.random.randint(-1,2,(N_row,N_col))
    gameboard.plot()

# Test get_max_action
def test_get_max_action():
    gameboard = GameBoard(N_row, N_col)
    agent = TDQNAgent(gameboard=gameboard)
    time_total = 0
    for _ in range(10000):
        gameboard.board = np.random.randint(-1,2,(N_row,N_col))
        start_time = time.time()
        index = agent.get_max_action()
        time_total += time.time() - start_time
        assert gameboard.board[index] == 0
    print("Passed agent.get_max_action()")
    print("Total time: ", time_total)

# Test get_max_action version
def test_get_max_action_slow():
    gameboard = GameBoard(N_row, N_col)
    agent = TDQNAgent(gameboard=gameboard)
    time_total = 0
    for _ in range(10000):
        gameboard.board = np.random.randint(-1,2,(N_row,N_col))
        start_time = time.time()
        index = agent.get_max_action_slow()
        time_total += time.time() - start_time
        assert gameboard.board[index] == 0
    print("Passed agent.test_get_max_action_slow()")
    print("Total time: ", time_total)

# Test get_max_action version
def test_get_max_action_slower():
    gameboard = GameBoard(N_row, N_col)
    agent = TDQNAgent(gameboard=gameboard)
    time_total = 0
    for _ in range(10000):
        gameboard.board = np.random.randint(-1,2,(N_row,N_col))
        start_time = time.time()
        index = agent.get_max_action_slower()
        time_total += time.time() - start_time
        assert gameboard.board[index] == 0
    print("Passed agent.test_get_max_action_slower()")
    print("Total time: ", time_total)

#test_plot_board()
test_fuctions = [
    test_get_max_action,
    test_get_max_action_slow,
    test_get_max_action_slower,
    test_get_random_action,
    #test_get_seq_reward,
    #test_move
]
testi = 0
print("--- Running tests ---")
start_time = time.time()
for test in test_fuctions:
    print("Running test {}/{}".format(testi+1, len(test_fuctions)))
    test()
    testi += 1
    pass
print("--- Passed all tests in %s seconds ---" % (time.time() - start_time))
