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

# test that the move sequence go to -1/0/1
n_episodes = 10
for idx, test in enumerate(test_moves):
    gameboard = GameBoard(N_row, N_col)
    agent = TDQNAgent(gameboard=gameboard, sync_target_episode_count=2)  # new agent per sequence

    for episodeIdx in range(n_episodes):
        print(f"[TRAINING] Sequence {idx+1}/{len(test_moves)}, Episode {episodeIdx+1}/{n_episodes}")
        
        for move in test:
            if move == test[-1]:
                agent.turn(forced_move=move)    # do the last move
                if expected_outcomes[idx] == 0:
                    # agent won't learn if the episode doesn't end, so need to end it myself
                    agent.gameboard.gameover = True
                agent.turn(forced_move=move)    # learn and restart
            else:
                agent.turn(forced_move=move)
        
    # done training, assert that the expected values are correct
    for move in test:
        with torch.no_grad():
            q_table = agent.qn(torch.reshape(torch.tensor(agent.gameboard.board*agent.gameboard.piece, dtype=torch.float64), (1,1,15,15)))
        expected_reward = q_table[0, move[0], move[1]].item()
        assert round(expected_reward) == expected_outcomes[idx] * agent.gameboard.piece
        gameboard.move(move[0], move[1])
    
        


