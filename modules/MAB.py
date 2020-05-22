from __future__ import division
import sys
import numpy as np
from scipy.stats import beta
import statsmodels.stats.proportion as ssp
from modules.utils import *



class ThompsonSampling:
	def __init__(self, K, alpha0, beta0, *args, **kwargs):
		assert alpha0 > 0,  "ValueError : alpha0 has to be strictly superior to 0."
		assert beta0 > 0,  "ValueError : beta0 has to be strictly superior to 0."
		self.alpha0 = alpha0
		self.beta0 = beta0
		self.alphas = np.ones(K) * alpha0
		self.betas = np.ones(K) * beta0
		self.rewards = np.array([])
		self.armsPlayed = []
		self.K = K
		self.t = 1
	
	def choose_arm(self):
		theta = np.random.beta(self.alphas, self.betas) # Sampling the Beta distribution of each arm
		self.t += 1
		return np.argmax(theta)
	
	def update(self, reward, armToPlay):
		self.rewards = np.append(self.rewards, reward)
		self.armsPlayed.append(armToPlay)
		if reward == 1:
		    self.alphas[armToPlay] += 1
		else:
		    self.betas[armToPlay] += 1

	def reset(self, hist=True, arm=-1):
		if arm == -1 :
			self.alphas = np.ones(self.K) * self.alpha0
			self.betas = np.ones(self.K) * self.beta0
		else :
			self.alphas[arm] = self.alpha0
			self.betas[arm] = self.beta0

		if hist == True:
			self.rewards = np.array([])
			self.armsPlayed = []
		self.t = 1
		

class UCB:
	def __init__(self, K, c, *args, **kwargs):
		self.ExpectedMeans = np.zeros(K)
		self.NbrPlayArms = np.ones(K)
		self.rewards = np.array([])
		self.armsPlayed = []
		self.c = c
		self.K = K
		self.t = 1

	def update(self, reward, armToPlay):
		self.rewards = np.append(self.rewards, reward)
		self.armsPlayed.append(armToPlay)
		self.ExpectedMeans[armToPlay] *=  self.NbrPlayArms[armToPlay]
		self.NbrPlayArms[armToPlay] += 1
		self.ExpectedMeans[armToPlay] += reward
		self.ExpectedMeans[armToPlay] /= self.NbrPlayArms[armToPlay]

	def choose_arm(self):
		ucb = self.ExpectedMeans + np.sqrt(self.c * np.log(self.t) / self.NbrPlayArms)
		self.t += 1
		return np.argmax(ucb)

	def reset(self, hist=True, arm=-1):
		if arm == -1 :
			self.ExpectedMeans = np.zeros(self.K)
			self.NbrPlayArms = np.ones(self.K)
		else:
			self.ExpectedMeans[arm] = 0.
			self.NbrPlayArms[arm] = 1.

		if hist == True:
			self.rewards = np.array([])
			self.armsPlayed = []
		self.t = 1


class BayesUCB(ThompsonSampling):
	def __init__(self, K, alpha0, beta0, HF, c, horizon, *args, **kwargs):
		super(BayesUCB, self).__init__(K, alpha0, beta0, *args, **kwargs)
		self.HF = HF
		self.c = c
		self.horizon = horizon
	
	def choose_arm(self):
		if self.HF:
		    T = self.t+1
		else:
			T = self.horizon
		d = 1/(self.t * np.log(T)**self.c)
		theta = beta.ppf(1 - d , self.alphas, self.betas)
		self.t += 1
		return np.argmax(theta)


class KLUCB(UCB):
	def __init__(self, K, horizon, c, HF, *args, **kwargs):
		super(KLUCB, self).__init__(K, c, *args, **kwargs)
		self.HF = HF
		self.horizon = horizon

	def SearchingKLUCBIndex(self):
		p = self.ExpectedMeans
		if self.HF:
		    T = self.t + 1 
		else:
			T = self.horizon
		d = (np.log(self.t) + self.c * np.log(np.log(T))) / self.NbrPlayArms
		return klIC(p,d)

	def choose_arm(self):
			indices = self.SearchingKLUCBIndex()
			self.t += 1
			return np.argmax(indices)


class CPUCB(UCB):
	def __init__(self, K, c, *args, **kwargs):
		super(CPUCB, self).__init__(K, c, *args, **kwargs)

	def choose_arm(self):
		_, ic = ssp.proportion_confint(np.floor(self.ExpectedMeans * self.NbrPlayArms),
										self.NbrPlayArms,
										1/ (self.t * np.log(self.t+1)**self.c),
										method = 'beta')
		self.t += 1
		return np.argmax(ic)


class MOSS(UCB):
	def __init__(self, K, c, horizon, *args, **kwargs):
		super(MOSS, self).__init__(K, c, *args, **kwargs)
		self.horizon = horizon

	def choose_arm(self):
		K = np.size(self.NbrPlayArms)
		moss = self.ExpectedMeans + np.sqrt(np.maximum(np.log(self.horizon / (K * self.NbrPlayArms)),0) / self.NbrPlayArms)
		self.t += 1
		return np.argmax(moss)


class DiscountedUCB(UCB):
	def __init__(self, K, c, gamma, *args, **kwargs):
		super(DiscountedUCB, self).__init__(K, c, *args, **kwargs)
		assert 0 < gamma <= 1,  "ValueError : The discount factor has to verify : 0 < gamma <= 1"
		self.gamma = gamma
		self.armsPlayedDiscounted = np.zeros(K)
		self.rewardsDiscounted = np.zeros(K)
	
	def update(self, reward, armToPlay):
		super(DiscountedUCB, self).update(reward, armToPlay)
		self.armsPlayedDiscounted *= self.gamma
		self.rewardsDiscounted *= self.gamma
		self.armsPlayedDiscounted[armToPlay] += 1
		self.rewardsDiscounted[armToPlay] += reward

	def choose_arm(self):
		n_t = np.sum(self.armsPlayedDiscounted)
		ucb = (self.rewardsDiscounted / self.armsPlayedDiscounted) + np.sqrt((self.c * np.log(n_t)) / (self.armsPlayedDiscounted))
		return np.argmax(ucb)


class SWUCB(UCB):
	def __init__(self, K, c, tau, *args, **kwargs):
		super(SWUCB, self).__init__(K, c, *args, **kwargs)
		assert tau >= 1, "ValueError: tau has to be superior or equal to 1."
		self.tau = tau
		self.lastRewards = np.zeros(tau) 
		self.lastArmsPlayed = np.full(tau, -1) 

	def update(self, reward, armToPlay):
		now = self.t % self.tau
		self.lastArmsPlayed[now] = armToPlay
		self.lastRewards[now] = reward
		super(SWUCB, self).update(reward, armToPlay)
		
	def choose_arm(self):
		evals = {arm : self.evaluate(arm) for arm in range(self.K)}
		return max(evals, key=evals.get)

	def evaluate(self, arm):
		lastChoices = np.count_nonzero(self.lastArmsPlayed == arm)
		return (np.sum(self.lastRewards[self.lastArmsPlayed == arm]) / lastChoices) + np.sqrt((self.c * np.log(min(self.t, self.tau))) / lastChoices)


class Exp3:
	def __init__(self, K, gamma, *args, **kwargs):
		assert 0 < gamma <= 1, "ValueError: gamma has to be in (0, 1]." 
		self.K = K
		self.t = 1
		self.gamma = gamma
		self.rewards = np.array([])
		self.armsPlayed = []
		self.weights = np.zeros(K)    ## S_t_i 
		self.probs = softmax(self.weights, scale = self.gamma)
	
	def update(self, reward, armToPlay):
		self.rewards = np.append(self.rewards, reward)
		self.armsPlayed.append(armToPlay)
		self.t += 1
		isPlayed = (np.arange(self.K) == armToPlay).astype(float)
		self.weights += 1 - isPlayed * (1 - isPlayed * reward) /  self.probs # S_t_i += 1 - (A_t == i)(1 - X_t_i) / (P_t_i)
		self.probs = softmax(self.weights, scale = self.gamma)
	
	def choose_arm(self):
		return np.random.choice(self.K, p=self.probs)

	def reset(self, hist=True, arm=-1):
		if arm == -1 :
			self.weights = np.zeros(K)    
		else :
			self.weights[arm] = 0.  
		self.probs = softmax(self.weights, scale = self.gamma)

		if hist == True:
			self.rewards = np.array([])
			self.armsPlayed = []
		self.t = 1
		

class Exp3S(Exp3):
	def __init__(self, K, gamma, *args, **kwargs):
		super(Exp3S, self).__init__(K, gamma, *args, **kwargs)
		self.weights = np.ones(K)
		self.probs = softmax(self.weights, scale = self.gamma)

	def update(self, reward, armToPlay):
		self.rewards = np.append(self.rewards, reward)
		self.armsPlayed.append(armToPlay)
		self.t += 1
		isPlayed = (np.arange(self.K) == armToPlay).astype(float)
		self.weights *= np.exp(self.gamma * isPlayed  * (reward / self.probs) / self.K)
		self.probs = (1 - self	.gamma) * self.weights / np.sum(self.weights) + self.gamma / self.K
	

