from __future__ import division
import numpy as np
from StochasticBandit_Modules import *


def ThompsonSampling(environment, horizon, alpha0, beta0):
    (alphas, betas, gainTS, armsPlayed) = TS_initialize(np.size(environment), alpha0, beta0)
    # Interaction with the environment
    for t in range(horizon):
        armToPlay = TS_recommendArm(alphas, betas)
        reward = int ((np.random.uniform() < environment[armToPlay]) == True)
        (alphas, betas, gainTS, armsPlayed) = TS_receiveReward(alphas, betas, reward, armToPlay, gainTS, armsPlayed)
    return armsPlayed


def BayesUCB(environment, horizon, alpha0, beta0, c, HF):
    # HF: Horizon free option.
    (alphas, betas, gainTS, armsPlayed) = TS_initialize(np.size(environment), alpha0, beta0)
    # Interaction with the environment
    for t in range(horizon):
        armToPlay = BayesUCB_recommendArm(alphas, betas, t+1, horizon, c, HF)
        reward = int ((np.random.uniform() < environment[armToPlay]) == True)
        (alphas, betas, gainTS, armsPlayed) = TS_receiveReward(alphas, betas, reward, armToPlay, gainTS, armsPlayed)
    return armsPlayed


def KLUCB(environment, horizon, c, HF):
    (ExpectedMeans, NbrPlayArms, gainUCB, armsPlayed) = UCB_initialize(np.size(environment))
    # Interaction with the environment
    for t in range(horizon):
        armToPlay = KLUCB_recommendArm(ExpectedMeans, NbrPlayArms, t+1, horizon, c, HF)
        reward = int ((np.random.uniform() < environment[armToPlay]) == True)
        (ExpectedMeans, NbrPlayArms, gainUCB, armsPlayed) = UCB_receiveReward(ExpectedMeans, NbrPlayArms, reward, armToPlay, gainUCB, armsPlayed)
    return armsPlayed


def CPUCB(environment, horizon, c):
    (ExpectedMeans, NbrPlayArms, gainUCB, armsPlayed) = UCB_initialize(np.size(environment))
    # Interaction with the environment
    for t in range(horizon):
        armToPlay = CPUCB_recommendArm(ExpectedMeans, NbrPlayArms, t+1, c)
        reward = int ((np.random.uniform() < environment[armToPlay]) == True)
        (ExpectedMeans, NbrPlayArms, gainUCB, armsPlayed) = UCB_receiveReward(ExpectedMeans, NbrPlayArms, reward, armToPlay, gainUCB, armsPlayed)
    return armsPlayed
