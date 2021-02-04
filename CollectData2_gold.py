import sveCacheSim as sim
import CacheModels
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm
from pathlib import Path

nruns = 1

data = {}

for trace in ['pennant_small','pennant_medium','pennant_large','lulesh_small','lulesh_medium','lulesh_large']:
    data[trace] = {}
    model = None
    data[trace]['base'] = [None]*nruns
    Path('touched/{}_{}.running'.format(trace, 'base')).touch()
    for i in range(nruns):
        data[trace]['base'][i],_ = sim.simulation(model, trace_id=trace, show_summary=False)

sim.save_object(data, 'DataV2/pennant_and_lulesh_base.pkl')

