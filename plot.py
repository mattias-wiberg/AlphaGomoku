import pandas as pd
import matplotlib.pyplot as plt
import torch

object = pd.read_pickle(r'moves_tots.p')
plt.plot(object)
print(len(object))
plt.waitforbuttonpress()