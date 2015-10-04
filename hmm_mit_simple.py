# Natural Language Toolkit: Hidden Markov Model
#
# Copyright (C) 2001-2006 University of Pennsylvania
# Author: Trevor Cohn <tacohn@csse.unimelb.edu.au>
#         Philip Blunsom <pcbl@csse.unimelb.edu.au>
#         Tiago Tresoldi <tiago@tresoldi.pro.br> (fixes)
#         Steven Bird <sb@csse.unimelb.edu.au> (fixes)
# URL: <http://nltk.sf.net>
# For license information, see LICENSE.TXT
#
# $Id: hmm.py 4048 2007-01-14 06:52:47Z stevenbird $

"""
Hidden Markov Models (HMMs) largely used to assign the correct label sequence
"""

from nltk.probability import *    # A====> Aller voir a http://www.nltk.org/_modules/nltk/probability.html
import numpy
from numpy import *

_NINF = float('-1e300')


class HiddenMarkovModel(object):
    """
    Hidden Markov model class, a generative model for labelling sequence data.
    These models define the joint probability of a sequence of symbols and
    their labels (state transitions) as the product of the starting state
    probability, the probability of each state transition, and the probability
    of each observation being generated from each state. This is described in
    more detail in the module documentation.
    
    This implementation is based on the HMM description in Chapter 8, Huang,
    Acero and Hon, Spoken Language Processing.
    """
    def __init__(self, symbols, states, transitions, outputs, priors):
        """
        Creates a hidden markov model parametised by the the states,
        transition probabilities, output probabilities and priors.

        @param  symbols:        the set of output symbols (alphabet)
        @type   symbols:        (seq) of any
        @param  states:         a set of states representing state space
        @type   states:         seq of any
        @param  transitions:    transition probabilities; Pr(s_i | s_j)
                                is the probability of transition from state i
                                given the model is in state_j
        @type   transitions:    C{ConditionalProbDistI}
        @param  outputs:        output probabilities; Pr(o_k | s_i) is the
                                probability of emitting symbol k when entering
                                state i
        @type   outputs:        C{ConditionalProbDistI}
        @param  priors:         initial state distribution; Pr(s_i) is the
                                probability of starting in state i
        @type   priors:         C{ProbDistI}
        """

        self._states = states
        self._transitions = transitions
        self._symbols = symbols
        self._outputs = outputs
        self._priors = priors


    def probability(self, sequence):
        """
        Returns the probability of the given symbol sequence. If the sequence
        is labelled, then returns the joint probability of the symbol, state
        sequence. Otherwise, uses the forward algorithm to find the
        probability over all label sequences.

        @return: the probability of the sequence
        @rtype: float
        @param sequence: the sequence of symbols which must contain the TEXT
            property, and optionally the TAG property
        @type sequence:  Token
        """
        return exp(self.log_probability(sequence))


    def _output_logprob(self, state, symbol):
        """
        @return: the log probability of the symbol being observed in the given
            state
        @rtype: float
        """
        return self._outputs[state].logprob(symbol)

 
    def best_path_simple(self, unlabeled_sequence):
        """
        Returns the state sequence of the optimal (most probable) path through
        the HMM. Uses the Viterbi algorithm to calculate this part by dynamic
        programming.  This uses a simple, direct method, and is included for
        teaching purposes.

        @return: the state sequence
        @rtype: sequence of any
        @param unlabeled_sequence: the sequence of unlabeled symbols 
        @type unlabeled_sequence: list
        """

        T = len(unlabeled_sequence)
        N = len(self._states)
        V = zeros((T, N), float64)
        B = {}

        # find the starting log probabilities for each state
        symbol = unlabeled_sequence[0]
        for i, state in enumerate(self._states):
            V[0, i] = self._priors.logprob(state) + \
                      self._output_logprob(state, symbol)
            B[0, state] = None

        # find the maximum log probabilities for reaching each state at time t
        for t in range(1, T):
            symbol = unlabeled_sequence[t]
            for j in range(N):
                sj = self._states[j]
                best = None
                for i in range(N):
                    si = self._states[i]
                    va = V[t-1, i] + self._transitions[si].logprob(sj)
                    if not best or va > best[0]:
                        best = (va, si)
                V[t, j] = best[0] + self._output_logprob(sj, symbol)
                B[t, sj] = best[1]

        # find the highest probability final state
        best = None
        for i in range(N):
            val = V[T-1, i]
            if not best or val > best[0]:
                best = (val, self._states[i])

        # traverse the back-pointers B to find the state sequence
        current = best[1]
        sequence = [current]
        for t in range(T-1, 0, -1):
            last = B[t, current]
            sequence.append(last)
            current = last
            #print current

        sequence.reverse()
        #print(sequence)
        return sequence

    def log_probability(self, unlabeled_sequence):
        """
        Return the forward probability matrix, a T by N array of
        log-probabilities, where T is the length of the sequence and N is the
        number of states. Each entry (t, s) gives the probability of being in
        state s at time t after observing the partial symbol sequence up to
        and including t.

        @param unlabeled_sequence: the sequence of unlabeled symbols 
        @type unlabeled_sequence: list
        @return: the forward log probability matrix
        @rtype:  array
        """
        T = len(unlabeled_sequence)
        N = len(self._states)
        alpha = zeros((T, N), float64)

        symbol = unlabeled_sequence[0]
        for i, state in enumerate(self._states):
            alpha[0, i] = self._priors.logprob(state) + \
                          self._outputs[state].logprob(symbol)
        for t in range(1, T):
            symbol = unlabeled_sequence[t]
            for i, si in enumerate(self._states):
                alpha[t, i] = _NINF
                for j, sj in enumerate(self._states):
                    alpha[t, i] = _log_add(alpha[t, i], alpha[t-1, j] +
                                           self._transitions[sj].logprob(si))
                alpha[t, i] += self._outputs[si].logprob(symbol)

        p = _log_add(*alpha[T-1, :])
        return p
        
    def __repr__(self):
        return '<HiddenMarkovModel %d states and %d output symbols>' \
                % (len(self._states), len(self._symbols))

class HiddenMarkovModelTrainer(object):
    """
    Algorithms for learning HMM parameters from training data. These include
    both supervised learning (MLE) and unsupervised learning (Baum-Welch).
    """
    def __init__(self, states=None, symbols=None):
        """
        Creates an HMM trainer to induce an HMM with the given states and
        output symbol alphabet. A supervised and unsupervised training
        method may be used. If either of the states or symbols are not given,
        these may be derived from supervised training.

        @param states:  the set of state labels
        @type states:   sequence of any
        @param symbols: the set of observation symbols
        @type symbols:  sequence of any
        """
        if states:
            self._states = states
        else:
            self._states = []
        if symbols:
            self._symbols = symbols
        else:
            self._symbols = []

 
    def train_supervised(self, labelled_sequences_X,  labelled_sequences_Y, **kwargs):
        """
        Supervised training maximising the joint probability of the symbol and
        state sequences. This is done via collecting frequencies of
        transitions between states, symbol observations while within each
        state and which states start a sentence. These frequency distributions
        are then normalised into probability estimates, which can be
        smoothed if desired.

        @return: the trained model
        @rtype: HiddenMarkovModel
        @param labelled_sequences: the training data, a set of
            labelled sequences of observations
        @type labelled_sequences: list
        @param kwargs: may include an 'estimator' parameter, a function taking
            a C{FreqDist} and a number of bins and returning a C{ProbDistI};
            otherwise a MLE estimate is used
        """

        # default to the MLE estimate
        estimator = kwargs.get('estimator')
        if estimator == None:
            estimator = lambda fdist, bins: MLEProbDist(fdist)

        # count occurences of starting states, transitions out of each state
        # and output symbols observed in each state
        starting = FreqDist()
        transitions = ConditionalFreqDist()
        outputs = ConditionalFreqDist()
        for i in range(len(labelled_sequences_X)):
            lasts = -1
            xs = labelled_sequences_X[i]
            ys = labelled_sequences_Y[i]
            #print len(xs)
            for j in range(len(xs)):
                state = ys[j]
                #print state
                #print lasts
                #print xs
                symbol = xs[j]
                if lasts == -1:
                    starting[state] +=1
                else:
                    transitions[lasts][state] +=1 
                outputs[state][symbol] +=1
                lasts = state

                # update the state and symbol lists
                if state not in self._states:
                    self._states.append(state)
                if symbol not in self._symbols:
                    self._symbols.append(symbol)

        # create probability distributions (with smoothing)
        N = len(self._states)
        pi = estimator(starting, N)
        #print pi
        A = ConditionalProbDist(transitions, ELEProbDist, N)
        B = ConditionalProbDist(outputs, ELEProbDist, len(self._symbols))
                               
        return HiddenMarkovModel(self._symbols, self._states, A, B, pi)

def _log_add(*values):
    """
    Adds the logged values, returning the logarithm of the addition.
    """
    x = max(values)
    if x > _NINF:
        sum_diffs = 0
        for value in values:
            sum_diffs += exp(value - x)
        return x + log(sum_diffs)
    else:
        return x    


