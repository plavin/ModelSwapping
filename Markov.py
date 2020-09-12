import random
from numba import njit
import numpy as np
from numpy.linalg import matrix_power

# Numba complains about using matrix_power
# TODO: Figure out if np.dot really would be faster
import warnings
warnings.filterwarnings('ignore')

@njit
def index2(array, item):
    for idx, val in enumerate(array):
        if val == item:
            return idx
    return len(array)-1


class MarkovModel:
    """
    This class can be used for training Markov chains and doing prediction
    
    Usage:
        (1) Create a new instance with mm = MarkovModel(N) where N is the number of states
        (2) Feed it a sequence with mm.add(n) where n is an integer in [0, N-1]
        (3) Have it calculate transition probabilities with mm.update_trainsition_matrix()
        (4) Begin using this mm for prediction with mm.get()
        (5) Continue adding more to the sequence with mm.add(n)
        
    """
    
    def __init__(self, N: np.int32):
        """
        Parameters
        ----------
        N: numpy.int32:
            The number of states for the model to use
        """
        
        self.nstates = N
        self.reset()
        
    def add(self, state: np.int32) -> None:
        """
        Add a state to the sequence being learned
        
        Parameters
        ----------
        state: numpy.int32:
            The next state in the sequence
        """
        
        if self.last_state == -1:
            self.last_state = state
            return
        self.count[self.last_state, state] += 1
        self.last_state = state
        if self.first_state == -1:
            self.first_state = state

    def reset(self):
        """
        Reset all data in for this Markov chain
        """
        
        self.trans = np.zeros((self.nstates,self.nstates), dtype=np.float32)
        self.count = np.ones((self.nstates,self.nstates), dtype=np.int64) #TODO: change back to zeros, but patch up nans later
        self.limit = np.zeros(self.nstates, dtype=np.float32)
        self.last_state = -1
        self.store_restrict = {}
        
        #MLE Stuff
        self.first_state = -1
        self.loglikelihood = 0
        self.bic = None
        
    def get_bic(self):
        """
        Get the Bayesian Information Criterion for this model
        """
        
        if self.bic is None:
            print('error: bic has not been calculated')
        return self.bic
    
    def peek_transition_matrix(self):
        """
        DEPRECATED: Get the the transition matrix without saving it
        """
        
        trans = np.zeros((self.nstates,self.nstates), dtype=np.float32)
        for i in range(self.nstates):
            trans[i,:] = self.count[i,:]/self.count[i,:].sum()
                
        return np.nan_to_num(trans, copy=False, nan=0.0)

    def update_transition_matrix(self):
        """
        Calculate the transition probabilities based on self.count and 
        store them in self.trans. These will be prefix-summed so that 
        it is easy to do prediction with the matrix.
        """
        
        # First, do that actual transition matrix, later we will prefix-sum it
        for i in range(self.nstates):
            self.trans[i,:] = self.count[i,:]/self.count[i,:].sum()
        
        limit_unsummed = matrix_power(self.trans, self.nstates)[0,:]
        self.limit = np.cumsum(limit_unsummed)
        self.limit[self.nstates-1] = 1.
        
        # MLE Calculation
        self.loglikelihood = np.log(limit_unsummed[self.first_state])
        tmp = np.log(self.trans)
        tmp[tmp == -np.inf] = 0
        self.loglikelihood += np.sum(
                                 np.multiply(
                                    tmp,
                                    self.count
                                 )
                              )
        self.bic = (self.nstates*self.nstates) * np.log(np.sum(self.count)) -  2 * self.loglikelihood
        
        # Prefix-sum for easier prediction
        for i in range(self.nstates):
            self.trans[i,:] = np.cumsum(self.count[i,:]/self.count[i,:].sum())
        
        # Need to reset the restriction matrices so they are recomputed
        self.store_restrict = {}
        
    def get(self, restrict=()):
        """
        Predict the next state in the sequence, only choosing from the states in restrict, 
        which should be a tuple of ints
        """
        
        r = random.random()
        #print(r)
        # The restrict parameter allows the user to give an allow list of states to return
        if restrict:
            if restrict in self.store_restrict: 
                if self.last_state == -1:
                    # we're not gonna worry about this for now. Just use the old code
                    # TODO: make this respect the restriction
                    return self.get()
                
                else:
                    self.last_state = restrict[index2(r < self.store_restrict[restrict][self.last_state,:],True)]
                    return self.last_state
            else:
                # update dict, memoize result
                res = self.count[:,restrict]
                temp = np.zeros(res.shape)
                for i in range(self.nstates):
                    temp[i,:] = np.cumsum(res[i,:]/res[i,:].sum())
                self.store_restrict[restrict] = temp
                return self.get(restrict)
        else:
            if self.last_state == -1:
                self.last_state = index2(r < self.limit, True)
                return self.last_state
            else:
                self.last_state = index2(r < self.trans[self.last_state,:], True)
                return self.last_state
        


if __name__ == "__main__":

    states = {0:'WH', 1:'WM', 2:'RH', 3:'RM'}
    nstates = len(states)

    print("We will generate a markov model with {} states, \n{}\n".format(nstates, states))

    prob = [.05, .1, .7, .15]
    print("First we will generate a random sequence of the states to train the model on. The states will ocurr with probabilies\n{} (respectively)\n".format(prob))
    prob = np.cumsum(prob)

    seq_len = 10000
    sequence = []

    for i in range(seq_len):
        sequence.append(index2(random.random() < prob, True))

    MM = MarkovModel(4)

    for s in sequence:
        MM.add(s)

    print("Now we train the model and get the transition matrix (which has already been np.cumsum'd for us")
    MM.update_transition_matrix()

    print(MM.trans)
    print("\n")

    pred = []

    for i in range(1000):
        pred.append(MM.get())

    print("We will now make a new sequence from the model and see if the probabbilities of each state match")

    print([pred.count(i) / len(pred) for i in range(4)])

