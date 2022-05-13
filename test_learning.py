from gameboard import GameBoard
from agent import TDQNAgent
import torch
N_row = 15
N_col = 15

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

n_episodes = 40
for idx, test in enumerate(test_moves):
    if expected_outcomes[idx] == 0:
        continue    # can't test against draw outcomes since the model only learns maximum rewards
    gameboard = GameBoard(N_row, N_col)
    agent = TDQNAgent(gameboard=gameboard)  # new agent per sequence

    for episodeIdx in range(n_episodes):
        print(f"[TRAINING] Sequence {idx+1}/{len(test_moves)}, Episode {episodeIdx+1}/{n_episodes}")
        
        # train on the sequence
        for move in test:
            agent.turn(forced_move=move)
            if agent.gameboard.gameover:    # some of the sequences have moves after the game has ended, so can't keep looping
                break
        agent.turn(forced_move=move)    # learn and reset the board
    
    # execute the sequence and assert that the expected reward for the winning sequence is 1 and -1 for the losing move
    for move_idx, move in enumerate(test):
        correct_expected_reward = expected_outcomes[idx] * agent.gameboard.piece
        if correct_expected_reward != 1:
            gameboard.move(move[0], move[1])
            continue    # can only test against winning moves since that is what the model learns

        with torch.no_grad():
            q_table = agent.qn(torch.reshape(torch.tensor(agent.gameboard.board*agent.gameboard.piece, dtype=torch.float64), (1,1,15,15)))
        expected_reward = round(q_table[0, move[0], move[1]].item())
        assert expected_reward == correct_expected_reward

        old_piece = gameboard.piece
        reward = gameboard.move(move[0], move[1])
        if agent.gameboard.gameover:
            assert expected_reward == reward * old_piece
            # undo the last two moves to assert that it knows the losing move reward is -1
            gameboard.board[move[0], move[1]] = 0
            losing_move = test[move_idx-1]
            gameboard.board[losing_move[0], losing_move[1]] = 0
            with torch.no_grad():
                q_table = agent.qn(torch.reshape(torch.tensor(agent.gameboard.board*agent.gameboard.piece, dtype=torch.float64), (1,1,15,15)))
            expected_reward = round(q_table[0, losing_move[0], losing_move[1]].item())
            assert expected_reward == -1
            break

    assert reward == expected_outcomes[idx]


