import sveCacheSim as sim
import CacheModels
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm

model = CacheModels.FixedHitRateSmartCache

#data, other = sim.simulation(model, 'small', show_summary=False)
data, other = sim.simulation(model, out_prefix='slu_steam1', trace_id='slu_steam1', show_summary=False)

print(data)

