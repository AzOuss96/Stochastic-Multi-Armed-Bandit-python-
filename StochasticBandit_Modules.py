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

def SearchingKLUCBIndex(ExpectedMeans, NbrPlayArms, t, horizon, c, HF):
    p = ExpectedMeans
    if HF:
        d = (np.log(t) + c*np.log(np.log(t+1)))/NbrPlayArms
    else:
        d = (np.log(t) + c*np.log(np.log(horizon)))/NbrPlayArms
    return klIC(p,d)


def plottingStochasticBandit(environment, armsPlayed, titre):
    horizon = np.size(armsPlayed)
    regret = np.max(environment)*np.ones(horizon) - environment[armsPlayed]
    regret = np.cumsum(regret)

    plt.plot(range(horizon),regret.tolist(),marker='.',label = titre)
    plt.legend(loc='lower right')
    plt.xlabel('Time step')
    plt.ylabel('cumulative regret')
    plt.show()


def TS_initialize(K, alpha0, beta0):
    if (alpha0 <=0 ):
        sys.exit('alpha0 must be > 0')
    if (beta0 <=0 ):
        sys.exit('beta0 must be > 0')
    alphas = np.ones(K)*alpha0
    betas = np.ones(K)*beta0
    gainTS = np.array([])
    armsPlayed = []
    return alphas, betas, gainTS, armsPlayed


def UCB_initialize(K):
    means = np.zeros(K)
    nbrPulls = np.ones(K)
    gainUCB = np.array([])
    armsPlayed = []
    return means, nbrPulls, gainUCB, armsPlayed

def UCB_receiveReward(ExpectedMeans, NbrPlayArms, reward, armToPlay, gain, armsPlayed):
    gain = np.append(gain, reward)
    armsPlayed.append(armToPlay)
    ExpectedMeans[armToPlay] *=  NbrPlayArms[armToPlay]
    NbrPlayArms[armToPlay] += 1
    ExpectedMeans[armToPlay] += reward
    ExpectedMeans[armToPlay] /= NbrPlayArms[armToPlay]
    return ExpectedMeans, NbrPlayArms, gain, armsPlayed


def UCB_recommendArm(ExpectedMeans, NbrPlayArms, t, c):
    K = np.size(NbrPlayArms)
    ucb = ExpectedMeans + np.sqrt(c*np.log(t)/NbrPlayArms)
    return np.argmax(ucb)


def TS_recommendArm(alphas, betas):
    theta = np.random.beta(alphas, betas) # Sampling the Beta distribution of each arm
    return np.argmax(theta)


def CPUCB_recommendArm(ExpectedMeans, NbrPlayArms, t, c):
    (e, ic) = ssp.proportion_confint(np.floor(ExpectedMeans*NbrPlayArms), NbrPlayArms, 1/(t*np.log(t+1)**c), method = 'beta')
    return np.argmax(ic)


def MOSS_recommendArm(ExpectedMeans, NbrPlayArms, horizon):
    K = np.size(NbrPlayArms)
    ucb = ExpectedMeans + np.sqrt(np.maximum(np.log(horizon/(K*NbrPlayArms)),0)/NbrPlayArms)
    return np.argmax(ucb)

def KLUCB_recommendArm(ExpectedMeans, NbrPlayArms, t, horizon, c, HF):
    indices = SearchingKLUCBIndex(ExpectedMeans, NbrPlayArms, t, horizon, c, HF)
    return np.argmax(indices)

def BayesUCB_recommendArm(alphas, betas,t, Horizon, c, HF):
    if HF:
        d = 1/(t*np.log(t+1)**c)
    else:
        d = 1/(t*np.log(Horizon)**c)
    theta = beta.ppf(1- d ,alphas, betas)
    return np.argmax(theta)

def TS_receiveReward(alphas, betas, reward, armToPlay, gainTS, armsPlayed):
    gainTS = np.append(gainTS, reward)
    armsPlayed.append(armToPlay)
    if reward == 1:
        alphas[armToPlay] += 1
    else:
        betas[armToPlay] += 1
    return alphas, betas, gainTS, armsPlayed

def PlayBernoulliArm(arm):
    return int ((np.random.uniform() < arm) == True)
