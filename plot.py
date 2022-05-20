from turtle import color
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from tqdm import tqdm

def moving_average(x, w):
    avg = []
    for i in tqdm(range(len(x))):
        avg.append(np.mean(x[max(0, int(i-w/2)):min(len(x)-1,int(i+w/2))]))
    return avg

save_path = 'networks/trained/'
network = 'deep_network_conv'
save_path += network

print('Loading data...')
moves_tots = pd.read_pickle(save_path+'/moves_tots.p')
wins = pd.read_pickle(save_path+'/wins.p')
#black_win_frac = pd.read_pickle(save_path+'/black_win_frac.p')
epsilons = pd.read_pickle(save_path+'/epsilons.p')
print('Done!')

print('Calculating black win fractions...')
black_win_frac = moving_average(np.array(wins)==-1, 100)
print('Done!')

print('Running moving average on moves_tots')
moves_tots_avg = moving_average(moves_tots, 1000)
print('Running moving average on wins')
wins_avg = moving_average(wins, 1000)

# Plotting
fig, ax1 = plt.subplots()
ax1.plot(moves_tots, label='Moves')
ax1.plot(moves_tots_avg, label='Average')
ax1.set_xlabel('Games')
ax1.set_ylabel('Number of Moves')

ax2 = ax1.twinx()
ax2.plot(wins_avg, label='Wins avg', color='green')
ax2.axhline(y=0, color='red', linestyle='--')
ax2.set_ylabel('Wins')
fig.tight_layout()

# Add high epsilon intervals
print('Adding high epsilon intervals...')
start = 0
end = 0
for i in tqdm(range(len(epsilons)-1)):
    if epsilons[i] <= 0.01 and epsilons[i+1] > 0.01: # From <= 0.01 to > 0.01
        start = i
    if epsilons[i] > 0.01 and epsilons[i+1] <= 0.01: # From > 0.01 to <= 0.01
        end = i
        plt.axvspan(start, end, color='red', alpha=0.5)
        start = 0
        end = 0
print('Done!')

print('Showing...')
plt.show()
