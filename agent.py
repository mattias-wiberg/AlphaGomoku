import torch
import copy
import pickle
import numpy as np
import random
from collections import namedtuple

Transition = namedtuple("Transition", 
                        ("old_state", "action_mask", "reward", "new_state", "terminal_mask", "legal_action_new_state_mask"))

class QN(torch.nn.Module):
    def __init__(self):
        super(QN, self).__init__()
        self.conv_1 = torch.nn.Conv2d(in_channels=1, out_channels=10, kernel_size=(5,5), stride=(1,1), dtype=torch.float64)
        self.pool_1 = torch.nn.MaxPool2d(kernel_size=(2,2), stride=(1,1))
        self.flatten_1 = torch.nn.Flatten()

        self.fc1 = torch.nn.Linear(1000, 1000, dtype=torch.float64)
        self.fc2 = torch.nn.Linear(1000, 225, dtype=torch.float64)
        self.fc3 = torch.nn.Linear(225, 225, dtype=torch.float64)
    
    def forward(self, x):
        out = self.flatten_1(self.pool_1(torch.relu(self.conv_1(x))))

        out = torch.relu(self.fc1(out))
        out = torch.relu(self.fc2(out))
        out = self.fc3(out)
        out = torch.reshape(out,(x.shape[0],15,15))
        return out

class TDQNAgent:
    def __init__(self,gameboard,alpha=0.001,epsilon=0.01,epsilon_scale=5000,replay_buffer_size=10000,batch_size=32,sync_target_episode_count=100,episode_count=10000):
        self.alpha=alpha
        self.epsilon=epsilon
        self.epsilon_scale=epsilon_scale
        self.replay_buffer_size=replay_buffer_size
        self.batch_size=batch_size
        self.sync_target_episode_count=sync_target_episode_count
        self.episode=0
        self.episode_count=episode_count
        self.reward_tots=[0]*episode_count
        self.gameboard = gameboard
        self.qn = QN()
        self.qnhat = copy.deepcopy(self.qn)
        self.exp_buffer = []
        self.criterion = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.qn.parameters(), lr=self.alpha)
        self.last_2_transitions = []

    def load_strategy(self,strategy_file):
        self.qn.load_state_dict(torch.load(strategy_file))
        self.qnhat = copy.deepcopy(self.qn)

    # Returns the row,col of a valid random action
    def get_random_action(self):
        mask = self.gameboard.board == 0
        index = np.unravel_index(np.argmax(np.random.random(mask.shape)*mask), mask.shape)
        return index

    # Returns the row,col of a valid action with the max future reward
    def get_max_action_slow(self):
        self.qn.eval()
        out = self.qn(torch.reshape(torch.tensor(self.gameboard.board, dtype=torch.float64), (1,1,15,15))).detach().numpy()[0]
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
        out = self.qn(torch.reshape(torch.tensor(self.gameboard.board, dtype=torch.float64), (1,1,15,15))).detach().numpy()[0]
        sorted_idx = np.argsort(out, axis=None)
        for idx in sorted_idx:
            idx = np.unravel_index(idx, out.shape) 
            if self.gameboard.board[idx] == 0:
                return idx
    
    def get_max_action_slower(self):
        self.qn.eval()
        out = self.qn(torch.reshape(torch.tensor(self.gameboard.board, dtype=torch.float64), (1,1,15,15))).detach().numpy()[0]
        mask = np.abs(self.gameboard.board) != 0 # Mask for valid actions
        ma = np.ma.masked_array(out, mask=mask)
        return np.unravel_index(np.ma.argmax(ma), (15,15))

    def select_action(self):
        if np.random.rand() < max(self.epsilon, 1-self.episode/self.epsilon_scale): # epsilon-greedy
            self.action = self.get_random_action()
        else: 
            self.action = self.get_max_action()
        
    def reinforce(self,batch):
        # batch: list of transitions
        # transition: tuple of old_states, action_mask, rewards, new_states, terminal_mask, legal_action_new_states_mask
        # TODO: fix this
        self.qn.train()
        self.qnhat.eval()
        old_states = batch[0]   # (32,1,15,15)
        action_mask = batch[1]  # (32,15,15)
        rewards = batch[2]      # (32)
        new_states = batch[3]   # (32,1,15,15)
        terminal_mask = batch[4]# (32)
        legal_action_new_states_mask = batch[5] # (32,15,15), torch.squeeze(new_states) == 0
        predictions = self.qn(old_states)[action_mask]
        with torch.no_grad():
            expected_future_reward = self.qnhat(new_states)
            expected_future_reward[legal_action_new_states_mask] = -np.infty
            targets = torch.max(torch.reshape(expected_future_reward, (32,15*15)), 1)[0]
            targets[terminal_mask] = 0
            targets += rewards
        loss = self.criterion(predictions,targets)
        self.optimizers.zero_grad()
        loss.backward()
        self.optimizers.step()

    def turn(self):
        if self.gameboard.gameover:
            self.episode+=1
            
            if self.episode%100==0:
                print('episode '+str(self.episode)+'/'+str(self.episode_count)+' (reward: ',str(np.mean(self.reward_tots[self.episode-100:self.episode])),')')
            
            if self.episode%1000==0:
                saveEpisodes=[1000,2000,5000,10000,20000,50000,100000,200000,500000,1000000];
                if self.episode in saveEpisodes:
                    torch.save(self.qn.state_dict(), 'qn.pth')
                    pickle.dump(self.reward_tots, open('reward_tots.p', 'wb'))
            
            if self.episode>=self.episode_count:
                return
            else:
                if (len(self.exp_buffer) >= self.replay_buffer_size) and ((self.episode % self.sync_target_episode_count)==0):
                    self.qnhat = copy.deepcopy(self.qn)
                self.gameboard.restart()
            
        else:
            self.select_action()

            transition = Transition()            
            transition.old_state = torch.as_tensor(self.gameboard.board, dtype=torch.float64)
            transition.action_mask = np.zeros((32,15,15), dtype=bool)
            transition.action_mask[self.action[0], self.action[1]] = 1

            reward = self.gameboard.move(self.action[0], self.action[1])
            reward = abs(reward) # Reward is always positive
            self.reward_tots[self.episode] += reward

            transition.reward = reward
            transition.terminal_mask = reward
            transition.new_state = torch.as_tensor(self.gameboard.board, dtype=torch.float64)
            transition.legal_action_new_state_mask = self.gameboard.board == 0
            self.last_2_transitions.append(transition)

            if len(self.last_2_transitions) == 3:
                self.exp_buffer.append(self.last_2_transitions.pop(0))
            if reward:
                # the previus move was a losing move
                self.last_2_transitions[0].reward = -1
                self.last_2_transitions[0].terminal_mask = 0

            if len(self.exp_buffer) >= self.replay_buffer_size:
                batch = random.sample(self.exp_buffer, k=self.batch_size)
                self.reinforce(batch)
                if len(self.exp_buffer) >= self.replay_buffer_size + 2:
                    self.exp_buffer.pop(0)
