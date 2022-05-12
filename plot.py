import pandas as pd
import torch

object = pd.read_pickle(r'moves_tots.p')
print(len(object))
for file in object:
    print(file)