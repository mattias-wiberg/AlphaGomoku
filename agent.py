import tensorflow as tf

class QN(tf.Module):
    def __init__(self, in_dim, out_dim):
        super(QN, self).__init__()
    
    def forward(self, x):
        return x

class Agent:
    # Agent for learning to play tetris using Q-learning
    def __init__(self,alpha,epsilon,epsilon_scale,replay_buffer_size,batch_size,sync_target_episode_count,episode_count):
        # Initialize training parameters
        self.alpha=alpha
        self.epsilon=epsilon
        self.epsilon_scale=epsilon_scale
        self.replay_buffer_size=replay_buffer_size
        self.batch_size=batch_size
        self.sync_target_episode_count=sync_target_episode_count
        self.episode=0
        self.episode_count=episode_count
        self.reward_tots=[0]*episode_count

    def init_game(self, gameboard):
        self.gameboard=gameboard
        self.qn = QN(gameboard.N_col*gameboard.N_row, 4*4)
        self.qnhat = tf.keras.models.clone_model(self.qn)
        self.exp_buffer = []
        self.criterion = tf.losses.MSELoss()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.alpha)


    def load_strategy(self, strategy_file):
        # Load strategy from file
        self.qn = tf.keras.models.load_model(strategy_file)
        self.qnhat = tf.keras.models.clone_model(self.qn)

    def read_state(self):
        self.state = self.gameboard.board.flatten()

    def select_action(self):
        out = self.qn(torch.tensor(self.state)).detach().numpy()
        if np.random.rand() < max(self.epsilon, 1-self.episode/self.epsilon_scale): # epsilon-greedy
            self.action = random.randint(0, (4*4)-1)
        else: 
            self.action = np.argmax(out)
        
        rotation = int(self.action / 4)
        position = self.action % 4
        self.gameboard.move()

    def fn_reinforce(self,batch):
        targets = []
        action_value = []
        self.qn.train()
        self.qnhat.eval()

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
        loss = self.criterion(action_value, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        # TO BE COMPLETED BY STUDENT
        # This function should be written by you
        # Instructions:
        # Update the Q network using a batch of quadruplets (old state, last action, last reward, new state)
        # Calculate the loss function by first, for each old state, use the Q-network to calculate the values Q(s_old,a), i.e. the estimate of the future reward for all actions a
        # Then repeat for the target network to calculate the value \hat Q(s_new,a) of the new state (use \hat Q=0 if the new state is terminal)
        # This function should not return a value, the Q table is stored as an attribute of self

        # Useful variables: 
        # The input argument 'batch' contains a sample of quadruplets used to update the Q-network

    def fn_turn(self):
        if self.gameboard.gameover:
            self.episode+=1
            if self.episode%100==0:
                print('episode '+str(self.episode)+'/'+str(self.episode_count)+' (reward: ',str(np.mean(self.reward_tots[self.episode-100:self.episode])),')')
            if self.episode%1000==0:
                saveEpisodes=[1000,2000,5000,10000,20000,50000,100000,200000,500000,1000000];
                if self.episode in saveEpisodes:
                    # TO BE COMPLETED BY STUDENT
                    # Here you can save the rewards and the Q-network to data files
                    torch.save(self.qn.state_dict(), 'qn_'+str(self.episode)+'.pth')
                    torch.save(self.qnhat.state_dict(), 'qnhat_'+str(self.episode)+'.pth')
                    pickle.dump(self.reward_tots, open('reward_tots_'+str(self.episode)+'.p', 'wb'))
            if self.episode>=self.episode_count:
                raise SystemExit(0)
            else:
                if (len(self.exp_buffer) >= self.replay_buffer_size) and ((self.episode % self.sync_target_episode_count)==0):
                    pass
                    # TO BE COMPLETED BY STUDENT
                    # Here you should write line(s) to copy the current network to the target network
                    self.qnhat = copy.deepcopy(self.qn)
                self.gameboard.fn_restart()
        else:
            # Select and execute action (move the tile to the desired column and orientation)
            self.fn_select_action()
            # TO BE COMPLETED BY STUDENT
            # Here you should write line(s) to copy the old state into the variable 'old_state' which is later stored in the ecperience replay buffer
            old_state = self.state.copy()

            # Drop the tile on the game board
            reward=self.gameboard.fn_drop()

            # TO BE COMPLETED BY STUDENT
            # Here you should write line(s) to add the current reward to the total reward for the current episode, so you can save it to disk later
            self.reward_tots[self.episode] += reward

            # Read the new state
            self.fn_read_state()
            
            # TO BE COMPLETED BY STUDENT
            # Here you should write line(s) to store the state in the experience replay buffer
            self.exp_buffer.append((old_state, self.action, reward, self.state.copy(), self.gameboard.gameover)) # Transition = {s_t, a_t, r_t, s_t+1}

            if len(self.exp_buffer) >= self.replay_buffer_size:
                # TO BE COMPLETED BY STUDENT
                # Here you should write line(s) to create a variable 'batch' containing 'self.batch_size' quadruplets 
                batch = random.sample(self.exp_buffer, k=self.batch_size)
                self.fn_reinforce(batch)
                if len(self.exp_buffer) >= self.replay_buffer_size + 2:
                    self.exp_buffer.pop(0) # Remove the oldest transition from the buffer
