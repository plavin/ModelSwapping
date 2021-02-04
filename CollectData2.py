import sveCacheSim as sim
import CacheModels
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm
from pathlib import Path

model_name = {CacheModels.FixedHitRateSmartCache:'FR',
              CacheModels.Markov4StateSmartCache: 'M4',
              CacheModels.Markov8StateSmartCache: 'M8',
              CacheModels.AllModelSmartCache: 'ALL'}
data = {}

for trace in ['pennant_small','pennant_medium','pennant_large','lulesh_small','lulesh_medium','lulesh_large']:
    data[trace] = {}
    for model in [CacheModels.FixedHitRateSmartCache,
                  CacheModels.Markov4StateSmartCache,
                  CacheModels.Markov8StateSmartCache,
                  CacheModels.AllModelSmartCache]:
        data = {}
        Path('touched/{}_{}.running'.format(trace, model_name[model])).touch()
        data[trace][model_name[model]][i],_ = sim.simulation(model, trace_id=trace, show_summary=False)

sim.save_object(data, 'DataV2/pennant_and_lulesh_2.pkl')

