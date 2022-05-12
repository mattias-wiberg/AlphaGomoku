import torch
import time
from agent import Transition
import numpy as np
import random

np.random.seed(12345)
torch.manual_seed(12345)
random.seed(12345)
n_moves = 100000

#### INIT ####
exp_buffer_list = []
for i in range(10000):
    transition = Transition(
        torch.as_tensor(np.random.randint(-1,2, (1,15,15)), dtype=torch.float64),
        np.random.randint(0,2,(15,15),dtype=bool),
        np.random.randint(-1,2),
        torch.as_tensor(np.random.randint(-1,2, (1,15,15)), dtype=torch.float64),
        np.random.randint(0,2,dtype=bool),
        np.random.randint(-1,2, (15,15)) == 0,
        )
    exp_buffer_list.append(transition)
    

old_states = torch.zeros((10000, 1, 15, 15))
action_masks = np.zeros((10000, 15, 15))
rewards = np.zeros(10000)
new_states = torch.zeros((10000, 1, 15, 15))
terminal_masks = np.zeros(10000)
illegal_action_new_state_mask = np.zeros((10000,15,15))
for i in range(10000):
    old_states[i] = torch.as_tensor(np.random.randint(-1,2, (1,15,15)), dtype=torch.float64)
    action_masks[i] = np.random.randint(0,2,(15,15),dtype=bool)
    rewards[i] = np.random.randint(-1,2)
    new_states[i] = torch.as_tensor(np.random.randint(-1,2, (1,15,15)), dtype=torch.float64)
    terminal_masks[i] = np.random.randint(0,2,dtype=bool)
    illegal_action_new_state_mask[i] = np.random.randint(-1,2, (15,15)) == 0
#### INIT ####


idx = 0
samples = random.sample(range(10000), 32)
old_states_batch = old_states[samples]
action_masks_batch = action_masks[samples]
rewards_batch = rewards[samples]
new_states_batch = new_states[samples]
terminal_masks_batch = terminal_masks[samples]
illegal_action_new_state_mask_batch = illegal_action_new_state_mask[samples]

#### BENCHMARK TENSORS ####
start_time = time.time()
for i in range(n_moves):
    old_states[idx] = torch.as_tensor(np.random.randint(-1,2, (1,15,15)), dtype=torch.float64)
    action_masks[idx] = np.random.randint(0,2,(1,15,15),dtype=bool)
    rewards[idx] = np.random.randint(-1,2)
    new_states[idx] = torch.as_tensor(np.random.randint(-1,2, (1,15,15)), dtype=torch.float64)
    terminal_masks[idx] = np.random.randint(0,2,dtype=bool)
    illegal_action_new_state_mask[idx] = np.random.randint(-1,2, (1,15,15)) == 0
    
    samples = random.sample(range(10000), 32)
    old_states_batch = old_states[samples]
    action_masks_batch = action_masks[samples]
    rewards_batch = rewards[samples]
    new_states_batch = new_states[samples]
    terminal_masks_batch = terminal_masks[samples]
    illegal_action_new_state_mask_batch = illegal_action_new_state_mask[samples]

    if idx == 9999:
        idx = 0
    else:
        idx += 1
    if i % 1000 == 0:
        print(f"{i/n_moves}")
print("--- All tensors: %s seconds ---" % (time.time() - start_time))
#### BENCHMARK TENSORS ####
# --- All tensors: 31.298457622528076 seconds ---

#### BENCHMARK LIST OF TENSORS ####
start_time = time.time()
for i in range(n_moves):
    del exp_buffer_list[0]
    transition = Transition(
        torch.as_tensor(np.random.randint(-1,2, (1,15,15)), dtype=torch.float64),
        np.random.randint(0,2,(15,15),dtype=bool),
        np.random.randint(-1,2),
        torch.as_tensor(np.random.randint(-1,2, (1,15,15)), dtype=torch.float64),
        np.random.randint(0,2,dtype=bool),
        np.random.randint(-1,2, (15,15)) == 0,
        )
    exp_buffer_list.append(transition)

    batch = random.sample(exp_buffer_list, 32)
    old_states_batch = torch.stack([x.old_state for x in batch])
    action_masks_batch = np.array([x.action_mask for x in batch])
    rewards_batch = np.array([x.reward for x in batch])
    new_states_batch = torch.stack([x.new_state for x in batch])
    terminal_masks_batch = np.array([x.terminal_mask for x in batch])
    illegal_action_new_state_mask_batch = np.array([x.illegal_action_new_state_mask for x in batch])
    if i % 1000 == 0:
        print(f"{i/n_moves}")
print("--- List of tensors: %s seconds ---" % (time.time() - start_time))
#### BENCHMARK LIST OF TENSORS ####
# --- List of tensors: 24.909301280975342 seconds ---

# Conclusion: lists are 20% faster

