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
# Extending This Code:
#  If you would like to add more cache models, you must do two things:
#   (1) Extend SwapModel with a model of your own. Look at FixedHitRateModel for an example.
#   (2) Create a cache that derives SmartCache and includes your new model as one
#       that it can train at run time. Look at FixedHitRateSmartCache for an example
#       of a cache that trains a single model for every phase, or look at
#       AllModelSmartCache for an example of a cache that trains multiple models.

import numpy as np
import importlib
import Markov
import enum
import json
import PhaseDetector
from random import random
from AppendArray import AppendArray

##############################
# Alternate Cache Models     #
# Must derive from SwapModel #
#                            #
# 1. Fixed Hit Rate          #
# 2. Markov 4-State          #
# 3. Markov 8-Sate           #
# 4. LSTM (Work-in-progress) #
##############################

class SwapModel():
    """
    This class serves as a template for other models that can be used to swap in in place of the
    base L1 cache. It should not be used directly.

    Classes inheriting from this should not change the function signatures.
    """

    def __init__(self):
        raise NotImplementedError()
    def access(self, params, isHit):
        raise NotImplementedError()
    def train(self):
        raise NotImplementedError()
    def peek_train(self):
        raise NotImplementedError()
    def get(self, params):
        raise NotImplementedError()
    def size(self, params):
        raise NotImplementedError()
    def complexity(self, params):
        raise NotImplementedError()
    def batch_access(self, params, isHit):
        for i in range(len(isHit)):
            par = {}
            for p in params:
                par[p] = params[p][i]
            self.access(params=par, isHit=isHit[i])

class FixedHitRateModel(SwapModel):
    def __init__(self):
        self.nhits     = np.uint64(0)
        self.naccesses = np.uint64(0)
        self.hit_rate  = np.float64(0.0)
        self.tmp_train = None

    def __str__(self):
        return 'FixedHitRateModel: (hit_rate: {})'.format(self.hit_rate)

    def access(self, params, isHit):
        self.naccesses += 1
        if isHit:
            self.nhits += 1

    def train(self):
        self.hit_rate = self.nhits / self.naccesses
        loglikelihood = self.nhits * np.log(self.hit_rate) + (self.naccesses - self.nhits) * np.log(1-self.hit_rate)
        self.bic = np.log(self.naccesses) - 2*(loglikelihood)

    def peek_train(self):
        if self.tmp_train is None:
            self.tmp_train = self.nhits / self.naccesses
            return None
        else:
            new =  self.nhits / self.naccesses
            diff = new - self.tmp_train
            self.tmp_train = new
            return diff

    def get(self, params):
        return random() < self.hit_rate

    def size(self):
        return 3*8 # 8 bytes nhits, 8 bytes naccesses, 8 bytes hit_rate

    def complexity(self):
        return 1 # Single comparison

class Markov4StateCacheModel(SwapModel):
    def __init__(self):
        self.model_states = {'RH':0, 'RM':1, 'WH':2, 'WM':3}
        self.rw_states    = {0:(0,1), 1:(2,3)}
        self.hit_states   = (0, 2)
        self.MM           = Markov.MarkovModel(len(self.model_states))
        self.tmp_trans    = None

    def __str__(self):
        return 'MarkovCacheModel:\ncounts:\n{}\ntrans:\n{}\nlimit:\n{})'.format(self.MM.count, self.MM.trans, self.MM.limit)

    def access(self, params, isHit):
        if params['isWrite']:
            if isHit:
                self.MM.add(self.model_states['WH'])
            else:
                self.MM.add(self.model_states['WM'])
        else:
            if isHit:
                self.MM.add(self.model_states['RH'])
            else:
                self.MM.add(self.model_states['RM'])

    def train(self):
        self.MM.update_transition_matrix()
        self.bic = self.MM.get_bic()

    def peek_train(self):
        if self.tmp_trans is None:
            self.tmp_trans = self.MM.peek_transition_matrix()
            return None
        else:
            new = self.MM.peek_transition_matrix()
            diff = np.linalg.norm(new - self.tmp_trans, ord='fro')
            self.tmp_trans = new
            return diff

    def get(self, params):
        next_state = self.MM.get(restrict=self.rw_states[params['isWrite']])
        return next_state in self.hit_states

    def size(self):
        return 3 * (4 * 4) * 8 # 3 4x4 matrices, count, transition prob, and restricted prob

    def complexity(self):
        return 1 # Single comparison

class Markov8StateCacheModel(SwapModel):
    def __init__(self):
        self.model_states = {'RHN':0, 'RMN':1, 'WHN':2, 'WMN':3,
                             'RHF':4, 'RMF':5, 'WHF':6, 'WMF':7}
        self.rw_states = {0:{0:(4,5), 1:(0,1)}, 1:{0:(6,7), 1:(2,3)}} # first index read/write, second near/far
        self.hit_states   = (0, 2, 4, 6)
        self.MM           = Markov.MarkovModel(len(self.model_states))
        self.granularity  = np.uint64(6)
        self.last_address = -1
        self.tmp_trans    = None

    def __str__(self):
        return 'Markov8StateCacheModel:\ncounts:\n{}\ntrans:\n{}\nlimit:\n{})'.format(self.MM.count, self.MM.trans, self.MM.limit)

    def access(self, params, isHit):
        if self.last_address == -1:
            self.last_address = params['addr']
            return

        isNear = (np.uint64(params['addr']) >> self.granularity) == (np.uint64(self.last_address) >> self.granularity)
        self.last_address = params['addr']

        if params['isWrite']:
            if isHit:
                if isNear:
                    self.MM.add(self.model_states['WHN'])
                else:
                    self.MM.add(self.model_states['WHF'])
            else:
                if isNear:
                    self.MM.add(self.model_states['WMN'])
                else:
                    self.MM.add(self.model_states['WMF'])
        else:
            if isHit:
                if isNear:
                    self.MM.add(self.model_states['RHN'])
                else:
                    self.MM.add(self.model_states['RHF'])
            else:
                if isNear:
                    self.MM.add(self.model_states['RMN'])
                else:
                    self.MM.add(self.model_states['RMF'])

    def train(self):
        self.MM.update_transition_matrix()
        self.bic = self.MM.get_bic()

    def peek_train(self):
        if self.tmp_trans is None:
            self.tmp_trans = self.MM.peek_transition_matrix()
            return None
        else:
            new = self.MM.peek_transition_matrix()
            diff = np.linalg.norm(new - self.tmp_trans, ord=2)
            self.tmp_trans = new
            return diff


    def get(self, params):
        isNear = (np.uint64(params['addr']) >> self.granularity) == (np.uint64(self.last_address) >> self.granularity)
        self.last_address = params['addr']

        next_state = self.MM.get(restrict=self.rw_states[params['isWrite']][isNear])
        return next_state in self.hit_states

    def size(self):
        return 3 * (8 * 8) * 8 # 3 8x8 matrices, count, transition prob, and restricted prob

    def complexity(self):
        return 2 # One comparison for isNear, another for hit

class LSTMCacheModel(SwapModel):
    def __init__(self):

        self.batch = 1
        self.ndelta = 16

        self.model = Sequential(
            [
                LSTM(1, activation='tanh',
                     stateful=True,
                     batch_size = self.batch,
                     input_shape=(None, self.ndelta)),
                #Dense(1, activation='sigmoid'),
            ]
        )

        self.model.compile(
            optimizer=keras.optimizers.RMSprop(),  # Optimizer
            loss='mse',
        )

        start_len = 2

        self.deltas_dtype = np.dtype([('d{}'.format(i+1), np.uint64) for i in self.ndelta])
        self.isHit_dtype = np.dtype([('isHit', np.bool_)])

        self.deltas = AppendArray(start_len, self.deltas_type)
        self.isHit = AppendArray(start_len, self.isHit_dtype)

        self.addr_history = collections.deque(maxlen=self.ndelta)
        [self.addr_history.append(0) for i in range(self.ndelta)]

        self.ntimes = 0

    def __str__(self):
        return 'LSTMModel'

    def access(self, params, isHit):
        deltas = params['addr'] - np.array([*self.addr_history], dtype=np.uint64)
        self.deltas.append(deltas)
        self.isHit.append(isHit)
        self.addr_history.appendleft(params['addr'])

    def train(self):
        self.deltas.shrink()
        self.isHit.shrink()
        gen = TimeseriesGenerator(self.deltas.data, self.isHit.data, self.batch, batch_size=self.batch)
        self.model.fit(gen, steps_per_epoch=100, epochs=30, verbose=0)

        # TODO: Use MSE to calculate BIC
        # https://machinelearningmastery.com/probabilistic-model-selection-measures/

    def get(self, params):
        deltas = params['addr'] - np.array([[[*self.addr_history]]], dtype=np.uint64)
        self.addr_history.append(params['addr'])

        self.ntimes+=1
        if (self.ntimes % 1000 == 0):
            print(self.ntimes)

        return self.model(deltas, training=False)[0][0] > .5

##################
# The base cache #
##################

class _Line:
    def __init__(self, config, startAddress):
        self.config = config
        self.startAddress = startAddress

    def hasLine(self, address):
        if address == self.startAddress:
            return True
        return False


class _Set:
    def __init__(self, config):
        self.config = config
        self.lines = [_Line(config, 0) for count in range(config.setSize)]

    def hasLine(self, lineAddress):

        # Check if the address is in this set
        for line in self.lines:
            if line.hasLine(lineAddress):
                return True

        for x in range(len(self.lines) - 1):
            # evict oldest line
            self.lines[x] = self.lines[x + 1]
        # add current line
        self.lines[len(self.lines) - 1] = _Line(self.config, lineAddress)

        return False

class BaseCache:
    use_simple_pd = False
    def __init__(self, config):
        self.nextLevel = None
        self.config = config
        self.sets = [_Set(self.config) for count in range(
            int(self.config.cacheSize / (self.config.setSize * self.config.lineSize * self.config.wordSize)))]
        self.evictCounter = 0
        self.prefEvictCounter = 0
        self.debug = False
        self.full_trace = False

        self.training = False


        self.trace_dtype = np.dtype([
          ('ip', np.uint64),
          ('addr', np.uint64),
          ('phase', np.int64),
          ('isWrite', np.bool_),
          ('isHit', np.bool_),
          ('state', np.uint64)
          ])

        self.naccesses = 0

        self.phase = -1
        self.phases=None

        self.cache_trace = AppendArray(10000, self.trace_dtype)
        self.full_trace = True

    def __str__(self):
        return 'Cache: ({})'.format(str(self.config))

    def real_cache_size(self):
        return 8192 #TODO do this properly later

    def real_cache_complexity(self):
        return 16 #TODO should be (associativity*2) as a linear search is needed to search and to evict

    #########################
    # Tracing functionality #
    #########################
    def get_full_trace(self):
        return self.cache_trace.get()

    def record(self, rec):
        self.cache_trace.append(rec)

    ############################
    # Phase detector callbacks #
    ############################

    # Store the phase for logging
    def phase_notify(self, p):
        self.phase = p


    ###############
    # Clear cache #
    ###############
    def reset(self):
        self.sets = [_Set(self.config) for count in range(
            int(self.config.cacheSize / (self.config.setSize * self.config.lineSize * self.config.wordSize)))]
        self.evictCounter = 0
        self.prefEvictCounter = 0

    ###################################
    # Functional cache implementation #
    ###################################
    def real_cache(self, address):
        lineAddress = address - (address % (self.config.lineSize * self.config.wordSize))
        return self.sets[lineAddress % len(self.sets)].hasLine(lineAddress)

    def hasAddress(self, address, prefetchEnabled, IP, isWrite = False):

        isHit = self.real_cache(address)

        if self.full_trace:
            self.record((IP, address, self.phase, isWrite, isHit, self.training))

        if isHit:
            return True, self.config.latency

        return self.cache_miss(prefetchEnabled, address, IP, isWrite)

    # sets next level
    def setNext(self, nextCache):
        self.nextLevel = nextCache

    # All children can call this on a miss to reduce code reuse
    # Should only be called by hasAddress
    def cache_miss(self, prefetchEnabled, address, IP, isWrite):
        if prefetchEnabled:
            self.prefEvictCounter += 1
        else:
            self.evictCounter += 1

        if self.nextLevel is None:
            return False, self.config.memLatency
        else:
            # go to next level, if available
            if prefetchEnabled:
                return self.nextLevel.hasAddress(address, 1, IP, isWrite)
            else:
                return self.nextLevel.hasAddress(address, 0, IP, isWrite)

##########################################
# The full cache swapping implementation #
##########################################

class PhaseState(enum.Enum):
    Training = 1
    Trained  = 2
    GiveUp   = 3

class CacheState(enum.Enum):
    Normal = 1
    Swapped = 2

class Phase():
    def  __init__(self, model_classes, state=PhaseState.Training):
        self.state = state
        self.models = [I() for I in model_classes]
        self.ntrain = 0 # phases used to train these models
        self.which_model = None # Once trained, the model that we chose as best
        self.score_history = AppendArray(10000,
                                          np.dtype([('model{}'.format(i), np.float32) for i in range(len(model_classes))]))

    def get(self, params):
        return self.models[self.which_model].get(params=params)


class SmartCache(BaseCache):
    use_simple_pd = True
    def __init__(self, config, models=None):

        # First, call the superclass's init
        super().__init__(config)

        self.phases = {-1:Phase([], PhaseState.GiveUp)}
        self.MAXTRAIN=3

        # Whether or not we have swapped in a trained model
        self.state = CacheState.Normal

        # Data collected from last phase that can be used for training
        # Should be reset at every call to phase_notify
        # Can reuse full trace data, most likely
        # IP, Addr, isWrite, isHit

        self.history_dtype = np.dtype([
          ('ip', np.uint64),
          ('addr', np.uint64),
          ('isWrite', np.uint64),
          ('isHit', np.uint64),
        ])

        self.predictions_dtype = np.dtype([
            ('model{}'.format(i), np.uint64) for i in range(len(models))
        ])

        self.history     = AppendArray(10000, self.history_dtype)
        self.predictions = AppendArray(10000, self.predictions_dtype)

        # Which models to train and evaluate
        self.models = models

        # Granularity to determine near/far, 6 is cache line
        self.granularity = np.uint64(6)
        self.last_address = 0

        self.near_hist_dtype = np.dtype([('isNear', np.uint64)])

        self.near_hist = AppendArray(10000, self.near_hist_dtype)


    # Train models, return -1 if we need more time to train this phase
    # All logic regarding how long to train a model and how to select a model should happen
    # here
    def model_trainer(self, phase, history, predictions, near_hist, transition):

        #acc, near_acc, sizes, complexity
        ideal = np.float64([2, 1, 0, 0])

        # Evaluate accuracy, if the last phase wasn't a transition
        if not transition:
            real = history['isHit']
            acc = []
            near_acc = []
            sizes = []
            scores = []
            complexity = []
            near_count_real = np.sum(real[near_hist==1]==0)

            for i in range(len(self.models)):
                pred = predictions['model{}'.format(i)]
                acc.append(np.sum(real == pred) / len(real))

                near_count_pred = np.sum(pred[near_hist==1]==0)
                if near_count_pred == 0:
                    if near_count_real == 0:
                        near_pct = 1
                    else:
                        near_pct = 0
                else:
                    near_pct = near_count_real / near_count_pred

                near_acc.append(near_pct)
                sizes.append(self.phases[phase].models[i].size() / self.real_cache_size())
                complexity.append(self.phases[phase].models[i].complexity() / self.real_cache_complexity())

            for i in range(len(self.models)):
                vec = np.float64([2*acc[i], near_acc[i], sizes[i], complexity[i]])
                scores.append(np.linalg.norm(ideal-vec))

            self.phases[self.phase].score_history.append(scores)

            if self.debug:
                print("acc: {}, phase{}".format(acc, self.phase))
                print("scores: {}, phase {}".format(scores, self.phase))



        if self.phases[phase].ntrain < self.MAXTRAIN:
            self.phases[phase].ntrain += 1
            for i in range(len(self.models)):
                par = {'IP':history['ip'], 'addr':history['addr'], 'isWrite':history['isWrite']}
                self.phases[self.phase].models[i].batch_access(params=par, isHit=history['isHit'])
                self.phases[self.phase].models[i].train()

        # Lets call it trained
        if self.phases[phase].ntrain >= self.MAXTRAIN and not transition:
            best = np.argmin(scores)

            self.phases[phase].which_model = best
            self.phases[phase].state = PhaseState.Trained

            #if acc[best] < .85:
            #    self.phases[phase].state = PhaseState.GiveUp
            #else:
            #    self.phases[phase].which_model = best
            #    self.phases[phase].state = PhaseState.Trained

        return self.phases[phase].state

    def predictor(self, address, IP, isWrite):
        pred = []
        for i in range(len(self.models)):
            pred.append(self.phases[self.phase].models[i].get(params={'IP':IP, 'addr':address, 'isWrite':isWrite}))
        self.predictions.append(pred)

    def notify_phase_init(self, phase):
        print('ERROR: Wrong PD used!')

    def notify_phase_start(self, phase):
        print('ERROR: Wrong PD used!')

    def notify_phase_stop(self, phase):
        print('ERROR: Wrong PD used!')

    def phase_notify(self, p):

        transition = (self.phase != p)
        self.phase = p

        # Grab the history and reset it at the end of every interval
        # Makes no assumptions about interval length
        history = self.history.get()
        self.history.reset()

        predictions = self.predictions.get()
        self.predictions.reset()

        near_hist = self.near_hist.get().squeeze()
        self.near_hist.reset()

        if p not in self.phases:
            self.phases[p] = Phase(self.models)

        # The new phase does not have a trained cache to swap in
        if self.phases[p].state is PhaseState.GiveUp:
            self.state = CacheState.Normal
            return

        # The new phase has already been trained. Swap it in.
        elif self.phases[p].state is PhaseState.Trained:
            self.state = CacheState.Swapped
            return

        # The new phase needs more time to train
        elif self.phases[p].state is PhaseState.Training:

            # Different trained cache was running, can't train on last interval. Swap it out
            # so we can train this interval
            if self.state is CacheState.Swapped:
                self.state = CacheState.Normal
                return

            # Base cache was running, let's use the data to train this model. If we did not
            # just transition, we can also evaluate the accuracy of the model
            elif self.state is CacheState.Normal:

                # Returns the state the the phase ends in after training
                ret = self.model_trainer(p, history, predictions, near_hist, transition)

                if ret is PhaseState.Trained:
                    self.state = CacheState.Swapped

            else:
                print("ERROR: Unhandled CacheState in parse_notify")

        else:
            print("ERROR: Unhandled PhaseState in parse_notify")


    def reset(self):
        super().reset()

    def swap_cache(self, addr, IP, isWrite):
        return self.phases[self.phase].get(params={'IP':IP, 'addr':addr, 'isWrite':isWrite})

    def hasAddress(self, address, prefetchEnabled, IP=None, isWrite = False):

        isNear = (np.uint64(address) >> self.granularity) == (np.uint64(self.last_address) >> self.granularity)
        self.last_address = address

        if self.state is CacheState.Normal:
            if self.phases[self.phase].state is PhaseState.Training:

                # TODO: run other models for evaluation purposes
                self.predictor(address, IP, isWrite)

                self.near_hist.append((isNear,))

            # Run base model
            isHit = self.real_cache(address)

        elif self.state is CacheState.Swapped:

            # Run trained model
            isHit = self.swap_cache(address, IP, isWrite)

        # Log into the history array
        self.history.append([IP, address, isWrite, isHit])

        if self.full_trace:
            self.record((IP, address, self.phase, isWrite, isHit, self.state.value))

        if isHit:
            return True, self.config.latency

        # Cache miss
        return self.cache_miss(prefetchEnabled, address, IP, isWrite)

class AccCache(SmartCache):
    use_simple_pd = True

    # Train models, return -1 if we need more time to train this phase
    # All logic regarding how long to train a model and how to select a model should happen
    # here
    def model_trainer(self, phase, history, predictions, near_hist, transition):

        #acc, near_acc, sizes, complexity
        ideal = np.float64([2, 1, 0, 0])

        # Evaluate accuracy, if the last phase wasn't a transition
        if not transition:
            real = history['isHit']
            acc = []
            near_acc = []
            sizes = []
            scores = []
            complexity = []
            near_count_real = np.sum(real[near_hist==1]==0)

            for i in range(len(self.models)):
                pred = predictions['model{}'.format(i)]
                acc.append(np.sum(real == pred) / len(real))

                near_count_pred = np.sum(pred[near_hist==1]==0)
                if near_count_pred == 0:
                    if near_count_real == 0:
                        near_pct = 1
                    else:
                        near_pct = 0
                else:
                    near_pct = near_count_real / near_count_pred

                near_acc.append(near_pct)
                sizes.append(self.phases[phase].models[i].size() / self.real_cache_size())
                complexity.append(self.phases[phase].models[i].complexity() / self.real_cache_complexity())

            for i in range(len(self.models)):
                scores.append(acc[i])

            self.phases[self.phase].score_history.append(scores)

            if self.debug:
                print("acc: {}, phase{}".format(acc, self.phase))
                print("scores: {}, phase {}".format(scores, self.phase))



        if self.phases[phase].ntrain < self.MAXTRAIN:
            self.phases[phase].ntrain += 1
            for i in range(len(self.models)):
                par = {'IP':history['ip'], 'addr':history['addr'], 'isWrite':history['isWrite']}
                self.phases[self.phase].models[i].batch_access(params=par, isHit=history['isHit'])
                self.phases[self.phase].models[i].train()

        # Lets call it trained
        if self.phases[phase].ntrain >= self.MAXTRAIN and not transition:
            best = np.argmax(scores)

            self.phases[phase].which_model = best
            self.phases[phase].state = PhaseState.Trained

            #if acc[best] < .85:
            #    self.phases[phase].state = PhaseState.GiveUp
            #else:
            #    self.phases[phase].which_model = best
            #    self.phases[phase].state = PhaseState.Trained

        return self.phases[phase].state

####################################################
# The SmartCache instantiated with specific models #
####################################################

class FixedHitRateSmartCache(SmartCache):
    def __init__(self, config):
        super().__init__(config, [FixedHitRateModel])

    def __str__(self):
        return 'FixedHitRateSmartCache: ({})'.format(str(self.config))

class Markov4StateSmartCache(SmartCache):
    def __init__(self, config):
        super().__init__(config, [Markov4StateCacheModel])

    def __str__(self):
        return 'Markov4StateSmartCache: ({})'.format(str(self.config))

class Markov8StateSmartCache(SmartCache):
    def __init__(self, config):
        super().__init__(config, [Markov8StateCacheModel])

    def __str__(self):
        return 'Markov8StateSmartCache: ({})'.format(str(self.config))

class AllModelSmartCache(SmartCache):
    def __init__(self, config):
        super().__init__(config, [FixedHitRateModel, Markov4StateCacheModel, Markov8StateCacheModel])

    def __str__(self):
        return 'AllModelSmartCache: ({})'.format(str(self.config))

class AllModelAccCache(AccCache):
    def __init__(self, config):
        super().__init__(config, [FixedHitRateModel, Markov4StateCacheModel, Markov8StateCacheModel])

    def __str__(self):
        return 'AllModelAccCache: ({})'.format(str(self.config))

#########################
# Utility functionality #
#########################

class CacheConfig:
    def __init__(self, cacheSize, lineSize, setSize, wordSize, latency, memLatency):
        self.cacheSize = cacheSize
        self.lineSize = lineSize
        self.setSize = setSize
        self.wordSize = wordSize
        self.latency = latency
        self.memLatency = memLatency
    def __str__(self):
        return 'CacheConfig: cacheSize: {}, lineSize: {}, setSize: {}, wordSize: {}, latency: {}, memLatency: {}'.format(self.cacheSize, self.lineSize, self.setSize, self.wordSize, self.latency, self.memLatency)

def parse_cache_model(model, swap={}):
    nlevels = 0
    cache_model = []
    fetch_data = {}

    with open(model) as json_data:
        data = json.load(json_data)

        nlevels = data[0]["nlevels"]
        if nlevels < 1:
            print('Error: the cache must have at least one level')
            exit()


        for lev in range(0, nlevels):

            # The cache config is the same no matter the type of model. It refers to
            # parameters of the functional model
            CC = CacheConfig(data[lev + 1]["cachesize"],
                             data[lev + 1]["linesize"],
                             data[lev + 1]["setsize"],
                             data[lev + 1]["wordsize"],
                             data[lev + 1]["latency"],
                             data[lev + 1]["memlatency"])

            # Use a different cache if specified in the swap dict
            CacheType = swap[lev] if lev in swap else BaseCache
            cache_model.append(CacheType(CC))

        fetch_data['fetch_level'] = data[len(data) - 1]["fetch_level"] - 1
        fetch_data['fetch_level_latency'] = data[len(data) - 1]["fetch_level_latency"] - 1

    return nlevels, cache_model, fetch_data

class CacheHierarchy:
    def __init__(self, modelfile, block_size = 10000, swap={}, binary_filename=None):
        nlevels, cache_model, fetch_data = parse_cache_model(modelfile, swap=swap)

        self.nlevels     = nlevels
        self.cache_model = cache_model
        self.fetch_data  = fetch_data

        self.pd = PhaseDetector.PhaseDetector(interval_len=block_size, stable_min=5, binary_filename=binary_filename)
        for i, cache in enumerate(cache_model):
            self.pd.register_listener(cache.phase_notify)

        for lev in range(0, self.nlevels):
            if lev != self.nlevels - 1:
                self.cache_model[lev].setNext(self.cache_model[lev + 1])

    def __str__(self):
        cache_str = ''
        for i in range(self.nlevels):
            cache_str = cache_str + '    L{}: '.format(i+1) + str(self.cache_model[i]) + '\n'
        pf_str = '  Prefetch level: {}'.format(self.fetch_data['fetch_level'])
        return 'CacheHierarchy:\n  Num Levels: {}\n{}{}'.format(self.nlevels, cache_str, pf_str)

class _AppendArrayGarbage():
    """
    A wrapper around a numpy array that supports the append operation, but does it
    without making copies. It does this by doubling the size of the array every time
    it fills up, amortizing the cost of appending.

    This is a utility class, and does not need to be used by users of the simulator.
    """

    def __init__(self, init_x, dt):
        """
        Parameters
        ----------

        init_x:
            The initial length of the array

        y:
            The width of the array (number of data per row)

        dt:
            The numpy data type for a row
        """

        self.data = np.zeros(init_x, dtype=dt)
        self.full = 0
        self.dt   = dt

    def append(self, val):
        """
        Append a row to the array.

        Parameters
        ----------

        val:
            The row (of type dt) to append to the array. The array will be expanded if needed.
        """

        val = np.array(tuple(val), dtype=self.dt)

        if self.full < len(self.data):
            self.data[self.full] = val
            self.full += 1
        else:
            self.data = np.lib.pad(self.data, (0,self.data.shape[0]),
                                   'constant', constant_values=(0))
            self.data[self.full] = val
            self.full += 1

    def shrink(self):
        """
        Shrink the array to the size of however many elements are in it.
        """

        self.data = self.data[:self.full]

    def get(self):
        """
        Get a copy of they numpy array associated with this _AppendArray
        """

        return self.data[:self.full].copy()

    def reset(self):
        """
        Reset the contents of this array
        """

        self.full = 0
