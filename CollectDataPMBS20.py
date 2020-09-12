import sveCacheSim as sim
import CacheModels
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm

nruns = 10

model_name = {CacheModels.FixedHitRateSmartCache:'FR',
              CacheModels.Markov4StateSmartCache: 'M4',
              CacheModels.Markov8StateSmartCache: 'M8',
              CacheModels.AllModelSmartCache: 'ALL'}
data = {}

for size in ['large']:
    data[size] = {}
    for model in [CacheModels.FixedHitRateSmartCache,
                  CacheModels.Markov4StateSmartCache,
                  CacheModels.Markov8StateSmartCache,
                  CacheModels.AllModelSmartCache]:
        data[size][model_name[model]] = [None]*nruns
        for i in range(nruns):
            data[size][model_name[model]][i],_ = sim.simulation(model, trace_id=size, show_summary=False)

sim.save_object(data, 'DataPMBS20/LargeAllModels10Runs.pkl')

