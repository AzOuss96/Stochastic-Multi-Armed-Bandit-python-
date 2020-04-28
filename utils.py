from __future__ import division
import numpy as np
from scipy.stats import beta
import matplotlib.pyplot as plt
import statsmodels.stats.proportion as ssp


def DivKL(p,q):
    eps = np.spacing(1)
    p = np.maximum(p, eps)
    p = np.minimum(p, 1-eps)
    q = np.maximum(q, eps)
    q = np.minimum(q, 1-eps)
    return  p*np.log(p/q) + (1-p)*np.log((1-p)/(1-q))

def klIC(p,d):
    lM = p.copy()
    uM = np.minimum(1, p + np.sqrt(d/2))
    for j in range(32):
        qM = (uM + lM)/2
        down = DivKL(p, qM) > d
        uM[down] = qM[down]
        lM[down == False] = qM[down == False]
    return uM

def play(MAB, environment, horizon):
    for t in range(horizon):
        armToPlay = MAB.choose_arm()
        reward = PlayBernoulliArm(environment[armToPlay])
        MAB.update(reward, armToPlay)
    return MAB.armsPlayed
	
def plottingStochasticBandit(environment, armsPlayed, titre):
    horizon = np.size(armsPlayed)
    regret = np.max(environment)*np.ones(horizon) - environment[armsPlayed]
    regret = np.cumsum(regret)

    plt.plot(range(horizon),regret.tolist(),marker='.',label = titre)
    plt.legend(loc='lower right')
    plt.xlabel('Time step')
    plt.ylabel('cumulative regret')
    plt.show()

def PlayBernoulliArm(arm):
    return int ((np.random.uniform() < arm) == True)

