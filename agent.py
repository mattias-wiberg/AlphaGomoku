import torch
import copy
import pickle
import numpy as np
import random

class QN(torch.nn.Module):
    def __init__(self):
        super(QN, self).__init__()
        self.conv_1 = torch.nn.Conv2d(in_channels=1, out_channels=10, kernel_size=(5,5), stride=(1,1),  dtype=torch.float64)
        self.pool_1 = torch.nn.MaxPool2d(kernel_size=(2,2), stride=(1,1))
        self.flatten_1 = torch.nn.Flatten()

        self.fc1 = torch.nn.Linear(1000, 500, dtype=torch.float64)
        self.fc2 = torch.nn.Linear(500, 500, dtype=torch.float64)
        self.fc3 = torch.nn.Linear(500, 15, dtype=torch.float64)
    
    def forward(self, x):
        out = self.flatten_1(self.pool_1(torch.relu(self.conv_1(x))))

        out = torch.relu(self.fc1(out))
        out = torch.relu(self.fc2(out))
        out = self.fc3(out)
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

    def load_strategy(self,strategy_file):
        self.qn.load_state_dict(torch.load(strategy_file))
        self.qnhat = copy.deepcopy(self.qn)

    # Returns the row,col of a valid random action
    def get_random_action(self):
        mask = self.gameboard.board == 0
        index = np.unravel_index(np.argmax(np.random.random(mask.shape)*mask), mask.shape)
        return index

    # Returns the row,col of a valid action with the max future reward
    def get_max_action(self):
        self.qn.eval()
        out = self.qn(torch.tensor(self.gameboard.board, dtype=torch.float64)).detach().numpy()
        mask = np.abs(self.gameboard.board) == 1 # Mask for non valid actions
        # Make the non valid cations have a lower value than the lowest valid action
        # this ensures that argmax will always give a valid action
        index = np.unravel_index(np.argmax(out-mask*(np.min(out)-1.0)), mask.shape) 
        return index

    def select_action(self):
        if np.random.rand() < max(self.epsilon, 1-self.episode/self.epsilon_scale): # epsilon-greedy
            self.action = self.get_random_action()
        else: 
            self.action = self.get_max_action()
        
    def reinforce(self,batch):
        # TODO: Implement this function correct with first and last player's perspective
        # batch[0]: old states (32, 15, 15, 1)
        targets = []
        action_value = []
        self.qn.eval()
        self.qnhat.eval()
        old_states = batch[0]
        self.qn(old_states)
        return 
        
        #predictions = 

        #targets = 

        for transition in batch:
            state = transition[0]
            action = transition[1]
            reward = transition[2]
            next_state = transition[3]
            terminal = transition[4]

            y = reward
            if not terminal:
                out = self.qnhat(torch.tensor(next_state)).detach().numpy()
                y += max(out)
            targets.append(torch.tensor(y, dtype=torch.float64))
            out = self.qn(torch.tensor(state))
            action_value.append(out[action])

        targets = torch.stack(targets)
        action_value = torch.stack(action_value)
        loss = self.criterion
        # Useful variables: 
        # The input argument 'batch' contains a sample of quadruplets used to update the Q-network

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
            # Select and execute action (move the peice to the desired column and orientation)
            self.select_action()

            # Copy the old state into the variable 'old_state' which is later stored in the ecperience replay buffer
            old_state = self.gameboard.board.copy()

            # Place the stone on the game board
            reward = self.gameboard.move(self.action[0], self.action[1])
            reward = abs(reward) # Reward is always positive

            self.reward_tots[self.episode] += reward

            # Store the state in the experience replay buffer
            self.exp_buffer.append((old_state, self.action, reward, self.state.copy(), self.gameboard.gameover)) # Transition = {s_t, a_t, r_t, s_t+1}

            if len(self.exp_buffer) >= self.replay_buffer_size:
                # TO BE COMPLETED BY STUDENT
                # Here you should write line(s) to create a variable 'batch' containing 'self.batch_size' quadruplets 
                batch = random.sample(self.exp_buffer, k=self.batch_size)
                self.reinforce(batch)
                if len(self.exp_buffer) >= self.replay_buffer_size + 2:
                    self.exp_buffer.pop(0) # Remove the oldest transition from the buffer
