import sveCacheSim as sim
import CacheModels
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm
from pathlib import Path
import sys
import os

model2name = {CacheModels.FixedHitRateSmartCache:'FR',
              CacheModels.Markov4StateSmartCache: 'M4',
              CacheModels.Markov8StateSmartCache: 'M8',
              CacheModels.AllModelSmartCache: 'ALL',
              CacheModels.AllModelAccCache: 'ACC',
              None:'BASE'}

name2model = {'FR':CacheModels.FixedHitRateSmartCache,
              'M4':CacheModels.Markov4StateSmartCache,
              'M8':CacheModels.Markov8StateSmartCache,
              'ALL':CacheModels.AllModelSmartCache,
              'ACC':CacheModels.AllModelAccCache,
              'BASE':None}

if len(sys.argv) < 3:
    print('error in num args')
    sys.exit()

trace = sys.argv[1]
model = sys.argv[2]

trace_list = ['meabo_small',
              'meabo_medium',
              'meabo_large',
              'pennant_small',
              'pennant_medium',
              'pennant_large',
              'lulesh_small',
              'lulesh_medium',
              'lulesh_large',
              'slu_steam1',
              'slu_steam2',
              'slu_orsirr_1',
              'slu_orsirr_2',
              'slu_orsreg_1',
              'blackscholes-simdev',
              'bodytrack-simdev',
              'ferret-simdev',
              'fluidanimate-simdev',
              'freqmine-simdev',
              'blackscholes-simsmall',
              'bodytrack-simsmall',
              'ferret-simsmall',
              'fluidanimate-simsmall',
              'freqmine-simsmall',
             ]

if trace not in trace_list:
    print('error in trace')
    sys.exit()

if model not in name2model.keys():
    print('error in model')
    sys.exit()

print('Running trace: {}, model: {}'.format(trace, model))

data = {}
Path('touched/{}-{}.running'.format(trace, model)).touch()
data,mod = sim.simulation(name2model[model], trace_id=trace, show_summary=False)

sim.save_object(data, '/storage/home/hhive1/plavin3/scratch/DataV5/{}-{}-data.pkl'.format(trace,model))
sim.save_object(mod,  '/storage/home/hhive1/plavin3/scratch/DataV5/{}-{}-model.pkl'.format(trace,model))

