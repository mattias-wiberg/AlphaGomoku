import torch
import pickle
import numpy as np
import random
from collections import namedtuple

Transition = namedtuple("Transition", 
                        ("old_state", "action_mask", "reward", "new_state", "terminal_mask", "illegal_action_new_state_mask"))

class QN(torch.nn.Module):
    def __init__(self):
        super(QN, self).__init__()
        self.conv_1 = torch.nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(5,5), stride=(1,1), dtype=torch.float64)
        self.flatten_1 = torch.nn.Flatten()

        self.fc0 = torch.nn.Linear(225, 225, dtype=torch.float64)
        self.flatten_0 = torch.nn.Flatten()

        self.fc1 = torch.nn.Linear(346, 346, dtype=torch.float64)
        self.fc2 = torch.nn.Linear(346, 225, dtype=torch.float64)
    
    def forward(self, x):
        out = self.flatten_1(torch.tanh(self.conv_1(x)))
        appended_board = torch.tanh(self.fc0(self.flatten_0(x)))
        
        out = torch.cat((out, appended_board), 1)

        out = torch.tanh(self.fc1(out))
        out = torch.tanh(self.fc2(out))
        out = torch.reshape(out,(x.shape[0],15,15))
        return out

class TDQNAgent:
    def __init__(self,gameboard,alpha=0.001,epsilon=0.01,epsilon_scale=5000,terminal_replay_buffer_size=10000,batch_size=32,sync_target_episode_count=10,episode_count=10000):
        self.alpha=alpha
        self.epsilon=epsilon
        self.epsilon_scale=epsilon_scale
        self.terminal_replay_buffer_size=terminal_replay_buffer_size
        self.batch_size=batch_size
        self.sync_target_episode_count=sync_target_episode_count
        self.episode=0
        self.episode_count=episode_count
        self.gameboard = gameboard
        self.qn = QN()
        self.qnhat = QN()
        self.qnhat.load_state_dict(self.qn.state_dict())
        self.terminal_buffer = []
        self.criterion = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.qn.parameters(), lr=self.alpha)
        self.last_2_transitions = []
        self.current_episode_buffer = []
        self.moves_tots = []

    def load_strategy(self,strategy_file):
        self.qn.load_state_dict(torch.load(strategy_file))
        self.qnhat.load_state_dict(self.qn.state_dict())     

    # Returns the row,col of a valid random action
    def get_random_action(self):
        mask = self.gameboard.board == 0
        index = np.unravel_index(np.argmax(np.random.random(mask.shape)*mask), mask.shape)
        return index

    # Returns the row,col of a valid action with the max future reward
    def get_max_action_slow(self):
        self.qn.eval()
        out = self.qn(torch.reshape(torch.tensor(self.gameboard.board*self.gameboard.piece, dtype=torch.float64), (1,1,15,15))).detach().numpy()[0]
        mask = np.abs(self.gameboard.board) == 0 # Mask for valid actions
        if out.min() <= 0: # Shift all valid actions positive > 0
            out[mask] += abs(out.min()) + 1.0
        # Make the non valid cations have a lower value than the lowest valid action
        # this ensures that argmax will always give a valid action
        index = np.unravel_index(np.argmax(out*mask), mask.shape)
        return index

    # Returns the row,col of a valid action with the max future reward
    def get_max_action(self):
        self.qn.eval()
        out = self.qn(torch.reshape(torch.tensor(self.gameboard.board*self.gameboard.piece, dtype=torch.float64), (1,1,15,15))).detach().numpy()[0]
        sorted_idx = np.flip(np.argsort(out, axis=None))
        for idx in sorted_idx:
            idx = np.unravel_index(idx, out.shape) 
            if self.gameboard.board[idx] == 0:
                return idx
    
    def get_max_action_slower(self):
        self.qn.eval()
        out = self.qn(torch.reshape(torch.tensor(self.gameboard.board*self.gameboard.piece, dtype=torch.float64), (1,1,15,15))).detach().numpy()[0]
        mask = np.abs(self.gameboard.board) != 0 # Mask for valid actions
        ma = np.ma.masked_array(out, mask=mask)
        return np.unravel_index(np.ma.argmax(ma), (15,15))

    def select_action(self):
        if np.random.rand() < max(self.epsilon, 1-self.episode/self.epsilon_scale): # epsilon-greedy
            self.action = self.get_random_action()
        else: 
            self.action = self.get_max_action()
        
    def reinforce(self, old_states_batch, action_masks_batch, rewards_batch, new_states_batch, terminal_masks_batch, illegal_action_new_state_mask_batch):
        self.qn.train()
        self.qnhat.eval()
        predictions = self.qn(old_states_batch)[action_masks_batch]
        with torch.no_grad():
            expected_future_reward = self.qnhat(new_states_batch)
            expected_future_reward[illegal_action_new_state_mask_batch] = -np.infty
            targets = torch.max(torch.reshape(expected_future_reward, (old_states_batch.shape[0],15*15)), 1)[0]
            targets[terminal_masks_batch] = 0
            targets += rewards_batch
        loss = self.criterion(predictions,targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def batch_and_reinforce(self, batch):
        old_states_batch = torch.stack([x.old_state for x in batch])
        action_masks_batch = torch.stack([x.action_mask for x in batch])
        rewards_batch = torch.tensor([x.reward for x in batch])
        new_states_batch = torch.stack([x.new_state for x in batch])
        terminal_masks_batch = torch.tensor([x.terminal_mask for x in batch])
        illegal_action_new_state_mask_batch = torch.stack([x.illegal_action_new_state_mask for x in batch])
        self.reinforce(old_states_batch, action_masks_batch, rewards_batch, new_states_batch, terminal_masks_batch, illegal_action_new_state_mask_batch)

    def turn(self, forced_move=None):
        # forced_move used in select_action for testing purposes
        if self.gameboard.gameover:
            self.episode+=1
            self.moves_tots.append(self.gameboard.n_moves)

            terminal_batch = random.sample(self.terminal_buffer, k=min(self.batch_size, len(self.terminal_buffer)))
            self.batch_and_reinforce(terminal_batch)
            self.batch_and_reinforce(self.current_episode_buffer)
            if len(self.terminal_buffer) >= self.terminal_replay_buffer_size + 2:
                self.terminal_buffer.pop(0)
            self.current_episode_buffer = []
            
            if self.episode%100==0:
                print('episode '+str(self.episode)+'/'+str(self.episode_count) + ' mean moves: '+str(np.mean(self.moves_tots[-100:])))
            
            if self.episode%1000==0:
                saveEpisodes=[1000,2000,5000,10000,20000,50000,100000,200000,500000,1000000];
                if self.episode % 1000 == 0:
                    pickle.dump(self.moves_tots, open('moves_tots.p', 'wb'))
                    torch.save(self.qn.state_dict(), 'qn.pth')
            
            if self.episode>=self.episode_count:
                SystemExit(0)
            else:
                if (self.episode % self.sync_target_episode_count)==0:
                    self.qnhat.load_state_dict(self.qn.state_dict()) 
                self.gameboard.restart()
            
        else:
            if forced_move is None:
                self.select_action()
            else:
                self.action = forced_move

            old_state = torch.reshape(torch.tensor(self.gameboard.board*self.gameboard.piece, dtype=torch.float64), (1,15,15))
            old_piece = self.gameboard.piece
            reward = self.gameboard.move(self.action[0], self.action[1])

            action_mask = torch.zeros((15,15), dtype=torch.bool)
            action_mask[self.action[0], self.action[1]] = 1
            transition = Transition(
                            old_state=old_state,
                            action_mask=action_mask,
                            reward=reward*old_piece,
                            new_state=torch.reshape(torch.tensor(self.gameboard.board*old_piece, dtype=torch.float64), (1,15,15)),
                            terminal_mask=self.gameboard.gameover,
                            illegal_action_new_state_mask=torch.tensor(self.gameboard.board != 0)
                            )
            self.current_episode_buffer.append(transition)

            if self.gameboard.gameover:
                # the second to last transition was a losing move
                self.current_episode_buffer[-2] = Transition(
                            old_state=self.current_episode_buffer[-2].old_state,
                            action_mask=self.current_episode_buffer[-2].action_mask,
                            reward=-1,    # CHANGED
                            new_state=self.current_episode_buffer[-2].new_state,
                            terminal_mask=self.gameboard.gameover,  # CHANGED
                            illegal_action_new_state_mask=self.current_episode_buffer[-2].illegal_action_new_state_mask
                            )
                self.terminal_buffer.append(self.current_episode_buffer[-2])
                self.terminal_buffer.append(self.current_episode_buffer[-1])
