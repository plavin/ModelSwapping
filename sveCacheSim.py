#!/usr/bin/python

# Copyright (c) 2019, Arm Limited and Contributors.
#
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# USAGE: 
#  You can run a simulation as simple as sveCacheSim.simulation(None), which 
#  will run a simulation with the default 3-level cache hierarchy, and use the 
#  default 'medium'-sized Meabo trace. To do some model swapping, use one 
#  of the objects derived from SmartCache in CacheModels.py, e.g.
#
#  stats, model = sveCacheSimm.simulation(CacheModels.Markov8StateSmartCache)
#
#

from collections import defaultdict
import os
import sys
import numpy as np
import pandas as pd
import time
import collections
import ntpath
import pickle
import PhaseDetector
from CacheModels import CacheHierarchy
          
def _run(trace, model, debug=False):
    """
    The actual simulation driver function. 
    
    This function drives the simulation, but is typically called by 
    simulation(), and not by other outside code. 
    
    Parameters:
    -----------
    
    trace:
        An object represting the trace to be simulated.
        
    model:
        The cache model used for simulation
        
    debug:
        Whether or not to print some debug output. (Default: False)
    """
    
    # Check that we have a Phase Detector
    if model.pd is None:
        raise AssertionError('No phase detector attached to model')
    
    # Get a place to store our stats about this run
    stats = Stats(model.nlevels)
    
    prog = ProgressBar(len(trace.VA))
    
    # Main simulation loop
    for idx,addr in enumerate(trace.VA):  
        # If no debugging, display a progressbar
        if not debug:
            prog.increment()

        # Inform the Phase Detector of the instruction pointer
        model.pd.access(trace.IP[idx])

        # Check the top level cache for address `addr`. If it doesn't
        # have it, it will automatically call the next level cache.
        isHit, latency = model.cache_model[0].hasAddress(
            addr, 0, IP=trace.IP[idx], isWrite=trace.isWrite[idx])  
        
        stats.total_cycles += latency
        if isHit:
            stats.total_hits += 1
            # Check which cache level got the hit based on latency
            for n in range(1, model.nlevels + 1):
                stats.accessCount[n - 1] += 1
                #if latency == data[n]["latency"]:
                if latency == model.cache_model[n-1].config.latency:
                    stats.hitcount[n - 1] += 1
                    break
        else:
            stats.total_misses += 1
            for n in range(1, model.nlevels + 1):
                stats.accessCount[n - 1] += 1

        
    if not debug:
        prog.finish()

    # Gather stats
    stats.total_accesses = len(trace.VA)
    for n in range(0, model.nlevels):
        stats.evictCount[n]     = model.cache_model[n].evictCounter
        stats.prefEvictCount[n] = model.cache_model[n].prefEvictCounter
        stats.hit_rate[n]       = 1.0 * stats.hitcount[n] / stats.accessCount[n]
        stats.miss_rate[n]      = 1.0 * (stats.accessCount[n] - stats.hitcount[n]) / stats.accessCount[n]
        stats.evict_rate[n]     = 1.0 * model.cache_model[n].evictCounter / stats.accessCount[n]
        stats.cache_trace[n]    = model.cache_model[n].get_full_trace()
        stats.phase_obj[n]      = model.cache_model[n].phases
        
    stats.phase_trace = model.pd.finalize()

    return stats

def simulation(newmodel, 
               out_prefix=None, 
               debug=False, 
               trace_id = 'medium',
               show_summary=True):
    """
    Run a simulation. 
    
    This method provides a wrapper around the run function, and does
    some things like load files for you, so you don't have to worry about where trace
    files are located, and where the gold stats are. 
    
    This is the preferred method of running simulations. 
    
    Parameters:
    -----------
    
    newmodel:
        An object deriving from CacheModels.SmartCache, or None, if you want to 
        run without model swapping.
        
    out_prefix:
        A prefix to use for storing the stats and model objects to files. If None, no files are
        written. 
        
    debug:
        Enable some debugging output. If false, a progress bar will be displayed instead. (Default: False)
        
    trace_id: 
        One of 'small', 'medium', and 'large', representing one of the three included Meabo traces. (Default: 'medium')
        
    show_summary:
        Whether or not to show a short summary of the stats as they compare to the
        stats in the gold files. (Default: True)
    """
    
    # These Meabo traces should come with your distribution of this code. 
    trace_list = {'small':'memtrace.meabo.i125.N512.P521.L1.log', 
                  'medium':'memtrace.meabo.i125.N512.P521.L2.log', 
                  'large':'memtrace.meabo.i125.N1536.P521.L3.log'}
    
    # These represet the stats for a run with no model swapping. 
    #gold_list = {'small':'gold/stats_abs_small.pkl', 
    #             'medium':'gold/base_medium_stats.pkl',
    #             'large':'gold/base_large_stats.pkl'}
    gold_list = {'small':'gold_python3.7/base_small_stats.pkl',
                 'medium':'gold_python3.7/base_medium_stats.pkl',
                 'large':'gold_python3.7/base_large_stats.pkl'}

    trace_file = 'traces/{}'.format(trace_list[trace_id])
    gold_file = gold_list[trace_id]
    
    model_file = 'base-models/3-level.json'
    if newmodel is not None:
        model = CacheHierarchy(model_file, swap={0:newmodel})
    else:
        model = CacheHierarchy(model_file)
        
    # Get the traace and run the simulation
    trace = traceToInts(trace_file, model, binary=True)
    stats = _run(trace, model, debug=debug)

    if show_summary:
        print('Trace: {}'.format(ntpath.basename(trace_file)))
        print_summary(stats, gold_file)
    
    # Optionally save the stats and model objects to pickle files
    if out_prefix is not None:
        stats_file = '{}_stats.pkl'.format(out_prefix)
        model_file = '{}_model.pkl'.format(out_prefix)
        save_object(stats, stats_file)
        save_object(model, model_file)
        print('Wrote stats to: {}'.format(stats_file))
        print('Wrote model to: {}'.format(model_file))

    return stats, model

###################
# Utility Classes #
###################

class Stats:
    """An object for collecting stats about a simulation run.
    
    Usage:
        (1) Initialize your Stats object and specify the number of cache levels you have.
            e.g. stats = Stats(3)
        (2) Fill it with data as you see fit.
    """
    
    def __init__(self, nlevels):
        """
        Parameters
        ----------
        
        nlevels:
            The number of cache levels you are collecting stats for.
        """
        
        self.nlevels         = nlevels
        self.hitcount        = [0] * nlevels
        self.accessCount     = [0] * nlevels
        self.evictCount      = [0] * nlevels
        self.prefEvictCount  = [0] * nlevels
        self.hit_rate        = [0] * nlevels
        self.miss_rate       = [0] * nlevels
        self.evict_rate      = [0] * nlevels
        self.total_hits      = 0
        self.total_misses    = 0
        self.total_cycles    = 0
        self.total_accesses  = 0
        self.cache_trace     = [0] * nlevels
        self.phase_trace     = None
        self.phase_stats     = [0] * nlevels
        self.phase_obj       = [0] * nlevels
        
    def print_stats(self, file=sys.stdout):
        """
        Print the stats for this object.
        WARNING - This does not print everything, for instance, it doesn't
        print any phase info.
        
        Parameters:
        -----------
        
        file:
            The file you would like to print the stats to (default: sys.stdout)
        """
        
        file.write("========\n{}\n========\n".format(run_name))
        for n in range(0, self.nlevels):
            file.write("l{} Hits\t\t{}\n".format(n + 1, self.hitcount[n]))
            file.write("l{} Accesses\t{}\n".format(n + 1, self.accessCount[n]))
            file.write("l{} Evicts\t{}\n".format( n + 1, self.evictCount[n]))
            file.write("l{} Hit Rate\t{:.2%}\n".format( n + 1, self.hit_rate[n]))
            file.write("l{} Miss Rate\t{:.2%}\n".format(n + 1, self.miss_rate[n]))
            file.write("l{} Evict Rate\t{:.2%}\n\n".format( n + 1, self.evict_rate[n]))
        file.write("Total Accesses\t{}\n".format(self.total_accesses))
        file.write("Total Hits\t{}\n".format(self.total_hits))
        file.write("Total Misses\t{}\n".format(self.total_misses))
        file.write("Total Cycles\t{}\n".format(self.total_cycles))


class ProgressBar:
    """
    A simple progress bar.
    
    Usage:
        (1) Initialize your progress bar by creating it and specifiying the number of iterations
            e.g. pbar = ProgressBar(200)
        (2) Call increment on the bar every iteration of your loop
            e.g. for i in range(200):
                    pbar.increment()
        (3) Call finish after your are done with your loop to display the total elapsed time
            e.g. pbar.finish()
    """
    
    def _minutes_string(seconds):
        """
        Converts seconds into mm:ss format and returns
        this as a string.
        
        Parameters
        ----------
        
        seconds:
            The values to be converted to minutes and seconds
        """
        
        minutes = int(seconds / 60)
        seconds = int(seconds - minutes * 60)
        return '{}:{:02}'.format(minutes, seconds)
    
    def __init__(self, maxidx, bar_length=40):
        """
        Parameters
        ----------
        
        maxidx:
            The number of loop iterations you will be doing
        
        bar_length:
            The length of the bar (in number of characters) (default 40)
        """
        
        self.bar_length = bar_length
        self.bar_progress = 0
        self.time_start = time.perf_counter()
        self.maxidx = maxidx
        self.idx = 0
        
    def increment(self):
        """
        Call this every iteration of your loop to update the progress bar.
            
        """
        
        if self.idx > self.bar_progress * self.maxidx/self.bar_length:
                time_now = time.perf_counter()
                time_per_it = (time_now - self.time_start) / self.idx
                time_left = (self.maxidx - self.idx) * time_per_it
                self.bar_progress += 1
                print('[' + 'x'*(self.bar_progress) + ' '*(self.bar_length-self.bar_progress) + ']'+ ' [ETA: {}]'.format(ProgressBar._minutes_string(time_left)) + ' '*10, end='\r')
                
        self.idx += 1
                
    def finish(self):
        """
        Call this after you are done with the progress bar.
        """
        
        print('[' + 'x'*(self.bar_length) + ']' + ' [Elapsed: {}]'.format(ProgressBar._minutes_string(time.perf_counter()-self.time_start)) + ' '*10)
        
        
class _Trace:
    """
    A class representing a trace of an application.
    """
    
    def __init__(self, IP, VA, isWrite):
        """
        Parameters:
        -----------
        
        IP:
            A list of instruction pointers
            
        VA:
            A list of virtual addresses
            
        isWrite:
            Whether or not each access is a write (True=>write, False=>read)
        """
        self.IP = IP # instruction pointer
        self.VA = VA # virtual address
        self.isWrite = isWrite
        
#####################
# Utility Functions #
#####################

def traceToInts(filename, model, zip_trace=False, binary=False):
    # TODO: actually do something with the sizes of the accesses
    dt = np.dtype([('isWrite', '?'), ('pad', 'S7'), ('VA', 'u8'), ('size', 'u8'), ('IP', 'u8')])
    data = np.fromfile(filename, dtype=dt)
    df = pd.DataFrame(data)
    return _Trace(df['IP'], df['VA'], df['isWrite'])

def save_object(obj, filename):
    with open(filename, 'wb') as output:  # Overwrites any existing file.
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)
        
def load_object(filename):
    with (open(filename, "rb")) as file:
        return pickle.load(file)
    
def print_summary(stats, gold):
    with (open(gold, "rb")) as file:
        stats_abs = pickle.load(file)
        
    print('Summary:')
        
    for i in range(len(stats.hit_rate)):
        print(' L{} percent change in hit count: {:.2f}%'.format(i+1, 100*(stats.hitcount[i]-stats_abs.hitcount[i])/stats_abs.hitcount[i]))
    
    pct = (stats.total_cycles - stats_abs.total_cycles) / stats_abs.total_cycles * 100
    print(' Percent change in cycles: {:.2f}%'.format(pct))
