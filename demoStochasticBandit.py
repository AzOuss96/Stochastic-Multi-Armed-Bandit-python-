from __future__ import division
import numpy as np
from StochasticBandit_Policies import * 
from utils import *
import matplotlib.pyplot as plt


"""
---------------------------------------------------------------------------------------------------------------------------
                                         Define the environment
---------------------------------------------------------------------------------------------------------------------------
"""

environment = np.array([0.7,0.1,0.1]) # Bernoulli distributions
horizon = 1000 # Overall number of interactions with the environment

"""
--------------------------------------------------------------------------------
                                      Creating the stochastic bandits
--------------------------------------------------------------------------------
"""

alpha0 = 1
beta0 = 1
c = 1
HF = True # Horizon free option for BayesUCB and KLUCB
K = np.size(environment)


ts = ThompsonSampling(K, alpha0, beta0)
ucb = UCB(K, c)
bayesucb = BayesUCB(K, alpha0, beta0, HF, c, horizon)
klucb = KLUCB(K, horizon, c, HF)
cpucb = CPUCB(K, c)
moss = MOSS(K, c, horizon)

"""
--------------------------------------------------------------------------------
                                      Launching the stochastic bandits
--------------------------------------------------------------------------------
"""

armsPlayedTS = play(ts, environment, horizon)
armsPlayedBayesUCB = play(bayesucb, environment, horizon)
armsPlayedKLUCB = play(klucb, environment, horizon)
armsPlayedCPUCB = play(cpucb, environment, horizon)
armsPlayedMOSS = play(moss, environment, horizon)
armsPlayedUCB = play(ucb, environment, horizon)

"""
------------------------------------------------------------------------------
                              Plotting results
-------------------------------------------------------------------------------
"""

plottingStochasticBandit(environment, armsPlayedTS, "Thompson Sampling")
plottingStochasticBandit(environment, armsPlayedBayesUCB, "Bayes UCB")
plottingStochasticBandit(environment, armsPlayedKLUCB, "KL UCB")
plottingStochasticBandit(environment, armsPlayedCPUCB, "CP UCB")
plottingStochasticBandit(environment, armsPlayedMOSS, "MOSS")
plottingStochasticBandit(environment, armsPlayedUCB, "UCB")
