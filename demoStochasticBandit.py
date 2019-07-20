from __future__ import division
import numpy as np
from StochasticBandit_Policies import * 
from StochasticBandit_Modules import *
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
                                      Launching the stochastic bandits
--------------------------------------------------------------------------------
"""

alpha0 = 1
beta0 = 1
c = 1
HF = True # Horizon free option for BayesUCB and KLUCB
armsPlayedTS = ThompsonSampling(environment, horizon, alpha0, beta0)
armsPlayedBayesUCB = BayesUCB(environment, horizon, alpha0, beta0, c, HF)
armsPlayedKLUCB = KLUCB(environment, horizon, c, HF)
armsPlayedCPUCB = CPUCB(environment, horizon, c)
armsPlayedMOSS = MOSS(environment, horizon)
armsPlayedUCB = UCB(environment, horizon, c)

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
