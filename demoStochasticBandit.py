from __future__ import division
import numpy as np
from StochasticBanditsPolicies import * 
from utils import *
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

sns.set()

"""
--------------------------------------------------------------------------------
                             Define the environment
--------------------------------------------------------------------------------
"""

horizon = 1000 # Overall number of interactions with the environment
environment = np.tile(np.array([0.7,0.1,0.1]), (horizon, 1)) # Bernoulli distributions

"""
--------------------------------------------------------------------------------
                       Creating the stochastic bandits
--------------------------------------------------------------------------------
"""

alpha0 = 1
beta0 = 1
c = 1
HF = True # Horizon free option for BayesUCB and KLUCB
K = environment.shape[1]


MABs = [ThompsonSampling(K, alpha0, beta0),
		UCB(K, c),
		BayesUCB(K, alpha0, beta0, HF, c, horizon),
		KLUCB(K, horizon, c, HF),
		CPUCB(K, c),
		MOSS(K, c, horizon)]

"""
--------------------------------------------------------------------------------
                      Launching the stochastic bandits
--------------------------------------------------------------------------------
"""

MAB_armsPlayed = {type(MAB).__name__ : play(MAB, environment, horizon) for MAB in MABs}

"""
------------------------------------------------------------------------------
                              Plotting results
-------------------------------------------------------------------------------
"""
fig, ax = plt.subplots(1, figsize=(15, 10))

for MAB, armsPlayed in MAB_armsPlayed.items():
	plottingStochasticBandit(environment,
							armsPlayed,
							title=MAB,
							ax=ax)

"""
------------------------------------------------------------------------------
                              Saving results
-------------------------------------------------------------------------------
"""
path = './plots/'
fig.savefig(os.path.join(path, 'Regrets'), dpi=100)


"""
------------------------------------------------------------------------------
                              Multiple runs
-------------------------------------------------------------------------------
"""
runs = 300
data = []

fig, ax = plt.subplots(1, figsize=(15, 10))

for MAB in MABs:
	data.append(playruns(MAB, environment, horizon, runs))
data = pd.concat(data)
data.to_csv('Regrets'+runs, index=False)

sns.lineplot(x='episod', y='cumulative regret', hue='bandit', data=data)
fig.savefig(os.path.join(path, 'Regrets'+str(runs)), dpi=100)
plt.show()

