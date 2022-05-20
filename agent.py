import torch
import pickle
import numpy as np
import random
from collections import namedtuple
import os
import sys
from networks.deep_network_conv import QN

Transition = namedtuple("Transition", 
                        ("old_state", "action_mask", "reward", "new_state", "terminal_mask", "illegal_action_new_state_mask"))

class TDQNAgent:
    def __init__(self,gameboard,save_path="networks/trained/"+"network/",alpha=0.00001,epsilon=0.01,epsilon_scale=4000,terminal_replay_buffer_size=10000,
                batch_size=32,sync_target_episode_count=10,episode_count=288000, device="cpu", re_exploration=40000, gamma=0.99):
        self.gamma = gamma
        self.alpha=alpha
        self.epsilon=epsilon
        self.epsilon_scale=epsilon_scale
        self.terminal_replay_buffer_size=terminal_replay_buffer_size
        self.batch_size=batch_size
        self.sync_target_episode_count=sync_target_episode_count
        self.episode=0
        self.episode_count=episode_count
        self.gameboard = gameboard
        self.save_path = save_path
        self.device = torch.device(device) 
        self.qn = QN(self.device)
        self.qnhat = QN(self.device)
        self.qnhat.load_state_dict(self.qn.state_dict())
        self.terminal_buffer = []
        self.criterion = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.qn.parameters(), lr=self.alpha)
        self.current_episode_buffer = []
        self.moves_tots = []
        self.wins = []
        self.black_win_frac = []
        self.epsilons = []
        #self.memory = []
        self.re_exploration = re_exploration

    def load_strategy(self, strategy_file, moves_tots_file, wins_file, black_win_frac_file, epsilons_file):
        if self.device == torch.device("cpu"):
            self.qn.load_state_dict(torch.load(strategy_file, map_location=torch.device("cpu")))
            self.qnhat.load_state_dict(self.qn.state_dict())
        else:
            self.qn.load_state_dict(torch.load(strategy_file))
            self.qnhat.load_state_dict(self.qn.state_dict())
        self.moves_tots = pickle.load(open(moves_tots_file,"rb"))
        self.wins = pickle.load(open(wins_file, "rb"))
        self.epsilons = pickle.load(open(epsilons_file, "rb"))
        self.black_win_frac = pickle.load(open(black_win_frac_file, "rb"))
        self.episode = len(self.moves_tots)

    # Returns the row,col of a valid random action
    def get_random_action(self):
        mask = self.gameboard.board == 0
        index = np.unravel_index(np.argmax(np.random.random(mask.shape)*mask), mask.shape)
        return index

    # Returns the row,col of a valid action with the max future reward
    def get_max_action_slow(self):
        self.qn.eval()
        out = self.qn(torch.reshape(torch.tensor(self.gameboard.board*self.gameboard.piece, dtype=torch.float64), (1,1,15,15))).detach().cpu().numpy()[0]
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
        out = self.qn(torch.reshape(torch.tensor(self.gameboard.board*self.gameboard.piece, dtype=torch.float64), (1,1,15,15))).detach().cpu().numpy()[0]
        #self.gameboard.set_out(out) # Set the output of the network to the gameboard
        self.gameboard.im2.set_data(out*self.gameboard.piece)
        sorted_idx = np.flip(np.argsort(out, axis=None))
        for idx in sorted_idx:
            idx = np.unravel_index(idx, out.shape) 
            if self.gameboard.board[idx] == 0:
                return idx
    
    def get_max_action_slower(self):
        self.qn.eval()
        out = self.qn(torch.reshape(torch.tensor(self.gameboard.board*self.gameboard.piece, dtype=torch.float64), (1,1,15,15))).detach().cpu().numpy()[0]
        mask = np.abs(self.gameboard.board) != 0 # Mask for valid actions
        ma = np.ma.masked_array(out, mask=mask)
        return np.unravel_index(np.ma.argmax(ma), (15,15))

    def select_action(self):
        if self.epsilon_scale != 0 and np.random.rand() < max(self.epsilon, 1-(self.episode%self.re_exploration)/self.epsilon_scale): # epsilon-greedy
            self.action = self.get_random_action()
        else: 
            self.action = self.get_max_action()
        
    def reinforce(self, old_states_batch, action_masks_batch, rewards_batch, new_states_batch, terminal_masks_batch, illegal_action_new_state_mask_batch):
        self.qn.train()
        self.qnhat.eval()
        predictions = self.qn(old_states_batch)[action_masks_batch]
        with torch.no_grad():
            expected_future_reward = self.qnhat(new_states_batch) * self.gamma
            expected_future_reward[illegal_action_new_state_mask_batch] = -np.infty
            targets = torch.max(torch.reshape(expected_future_reward, (old_states_batch.shape[0],15*15)), 1)[0]
            targets[terminal_masks_batch] = 0
            targets += rewards_batch.to(self.device)
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
            if self.epsilon_scale != 0:
                self.epsilons.append(max(self.epsilon, 1-(self.episode%self.re_exploration)/self.epsilon_scale))
            else:
                self.epsilons.append(self.epsilon)
            self.moves_tots.append(self.gameboard.n_moves)
            self.black_win_frac.append(self.wins[-100:].count(-1) / 100)
            self.episode+=1

            terminal_batch = random.sample(self.terminal_buffer, k=min(self.batch_size, len(self.terminal_buffer)))
            self.batch_and_reinforce(terminal_batch)
            self.batch_and_reinforce(self.current_episode_buffer)
            #memory_batch = random.sample(self.memory, k=min(self.batch_size, len(self.memory)))
            #self.batch_and_reinforce(memory_batch)
            if len(self.terminal_buffer) >= self.terminal_replay_buffer_size + 2:
                self.terminal_buffer.pop(0)
            #if len(self.memory) >= self.terminal_replay_buffer_size + 2:
            #    self.memory.pop(0)
            self.current_episode_buffer = []
            
            if self.episode%100==0:
                print(f'[{self.episode}/{self.episode_count}] mean moves: {np.mean(self.moves_tots[-100:])}, black win fraction: {round(self.black_win_frac[-1], 2)}, epsilon: {max(self.epsilon, 1-(self.episode%self.re_exploration)/self.epsilon_scale)}')
            
            if self.episode % 1000 == 0:
                if not os.path.exists(self.save_path):
                    os.makedirs(self.save_path)
                pickle.dump(self.moves_tots, open(self.save_path +'moves_tots.p', 'wb'))
                pickle.dump(self.wins, open(self.save_path +'wins.p', 'wb'))
                pickle.dump(self.black_win_frac, open(self.save_path +'black_win_frac.p', 'wb'))
                pickle.dump(self.epsilons, open(self.save_path +'epsilons.p', 'wb'))
                torch.save(self.qn.state_dict(), self.save_path +'qn.pth')
            
            if self.episode>=self.episode_count:
                sys.exit()
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
            if self.gameboard.gameover:
                self.wins.append(reward)

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
            #self.memory.append(transition)

            if self.gameboard.gameover:
                # the second to last transition was a terminal move (loss/tie)
                self.current_episode_buffer[-2] = Transition(
                            old_state=self.current_episode_buffer[-2].old_state,
                            action_mask=self.current_episode_buffer[-2].action_mask,
                            reward=-1*abs(reward),    # CHANGED
                            new_state=self.current_episode_buffer[-2].new_state,
                            terminal_mask=self.gameboard.gameover,  # CHANGED
                            illegal_action_new_state_mask=self.current_episode_buffer[-2].illegal_action_new_state_mask
                            )
                #self.memory[-2] = self.current_episode_buffer[-2]
                self.terminal_buffer.append(self.current_episode_buffer[-2])
                self.terminal_buffer.append(self.current_episode_buffer[-1])
