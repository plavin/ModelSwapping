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

import BitVector.BitVector as bitvec
import numpy
from typing import Callable as _callable

def _similarity(sig1: bitvec, sig2: bitvec):
    xor_bits = sig1 ^ sig2
    or_bits  = sig1 | sig2
    return xor_bits.count_bits() / or_bits.count_bits()

class PhaseTrace:
    """
    A class encapsulating a phase trace as well as some stats about that trace.
    
     Attributes
     ----------
     trace:
         The list of the phase id for each interval
     
     nphases:
         The number of unique phases encountered
     
     phase_count:
         A list of length nphases where phase_count[i] is the number of intervals assigned to phase i
         
     phase_unique:
         A list of length nphases where phase_unique[i] is the number of times phase i was entered

    """
    
    def __init__(self, trace):
        self.trace = trace.copy()
        self.nphases = max(trace)+1
        self.phase_count = [trace.count(i) for i in range(self.nphases)]
        for i in range(len(trace)-1):
            if trace[i] == trace[i+1]:
                trace[i] = -1
        self.phase_unique = [trace.count(i) for i in range(self.nphases)]


class PhaseDetector:
    """
    A class for phase detection using working set analysis.
    
    Usage: 
        (1) Create an instance of the PhaseDetector class, e.g.
                pd = PhaseDetector.PhaseDetector()
                
        (2) Register callbacks for listeners with register_listeners(), e.g.
                def callback(phase: numpy.int64):
                    print('PD Notification: last interval had phase id ', phase)
                pd.register_listeners('Listner1', callback)
         
        (3) As you run your simulation, send all IPs to the phase detector using access(), e.g.
                for ip in ip_list:
                    pd.access(ip)
                    
        (4) When you simulation has ended, access aggregate stats with finalize(), e.g.
                stats = pd.finalize()

    """
    
    def __init__(self, interval_len=10000, stable_min=4, threshold=.5, bits_log2 = 10, drop_bits = 3):
        """
       
        Parameters
        ----------
        interval_len : numpy.uint64 
            The length of an interval in number of instructions (default 10000)
            
        stable_min : numpy.uint64
            The number of intervals with each signature similar to the previous one
            for a phase to be identified (default 4)
            
        threshold : numpy.float32 
            The minimum similarity allowed for two intervals to be similar (default 0.5)
            
         bits_log2 : numpy.uint32
            Log 2 of the size of the bit vector used for the signature (default 10)
        
        drop_bits : numpy.uint32
            The number of bits to drop from the right side of an IP before hashing (default 3)
        """
        
        self.interval_len = numpy.uint64(interval_len)
        self.stable_min   = numpy.uint64(stable_min)
        self.threshold    = numpy.float32(threshold)
        self.bits_log2    = numpy.uint32(bits_log2)
        self.drop_bits    = numpy.uint32(drop_bits)
        
        # Use the reset functions to populate instance variables
        self.reset()
        self.reset_listeners()
        
    def _hash_addr(self, x: numpy.uint64):
        x = numpy.uint64(x) >> self.drop_bits
        return hash(str(x)) >> (64 - self.bits_log2)
    
    def _new_sig(self):
        return bitvec(bitlist=[0]*(2**self.bits_log2))
    
    def reset(self):
        """Resets the state of the phase detector but does not deregister 
        any of the listeners.
        """
        
        self.phase_table = []
        self.sig         = self._new_sig()
        self.last_sig    = self._new_sig()
        self.stable      = numpy.uint64(0)
        self.phase_trace = []
        self.naccesses   = numpy.uint64(0)
        self.phase       = numpy.int64(-1)
        
    def register_listener(self, function: _callable[[numpy.int64], None]) -> None:
        """Adds an instruction pointer to the signature, and, if this is the 
        end of a phase, notifies any listeners of the phase.
        
        Parameters
        ----------
        function:
            A callback that takes the phase id (an numpy.int64) as input
        """
        self.listeners.append(function)
        
    def reset_listeners(self) -> None:
        """Removes any listeners.
        """
        self.listeners = []
        
    def finalize(self) -> PhaseTrace:
        """Returns the phase and some basic stats in a PhaseTrace object.
        """
        return(PhaseTrace(self.phase_trace))
    
    def access(self, ip: numpy.uint64) -> None:
        """Adds an instruction pointer to the signature, 
        and, if this is the end of a phase, notifies
        any listeners of the phase.
        
        Parameters
        ----------
        ip : numpy.uint64
            The instruction pointer to add to the signature
        """
        
        # Update the signature
        self.sig[self._hash_addr(ip)] = 1
        
        # If we have reached the end of a block, see what the signature was, and identify the phase
        if self.naccesses % self.interval_len == 0 and self.naccesses != 0: 
            
            sig_difference = _similarity(self.sig, self.last_sig)
            if sig_difference < self.threshold:
                self.stable = self.stable + 1
                
                # If we have reached stablility and we dont know our phase then we are in a new phase
                if self.stable >= self.stable_min and self.phase == -1:
                    self.phase_table.append(self.sig)
                    self.phase = len(self.phase_table) - 1
                        
            else:
                self.stable = numpy.uint64(0)
                self.phase = numpy.int64(-1)
                
                # Either we weren't in a phase or one just ended. Check to see if we are in a phase we recognize.
                if len(self.phase_table) > 0:
                    tmp_similarity = [_similarity(self.sig, self.phase_table[i]) for i in range(len(self.phase_table))]
                    best_match = tmp_similarity.index(min(tmp_similarity))
                    if tmp_similarity[best_match] < self.threshold:
                        self.phase = best_match

            self.last_sig = self.sig.deep_copy()
            self.sig = self._new_sig()
            self.phase_trace.append(self.phase)
            
            # Notify listeners of the phase assigned to the last interval
            for callback in self.listeners:
                callback(self.phase) 

        self.naccesses = self.naccesses + 1
        return None