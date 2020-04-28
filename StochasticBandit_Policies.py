from __future__ import division
import numpy as np
#from StochasticBandit_Modules import *
from __future__ import division
import numpy as np
from scipy.stats import beta
import statsmodels.stats.proportion as ssp
from utils import DivKL, klIC



class ThompsonSampling:
	def __init__(self, K, alpha0, beta0):
		if (alpha0 <=0 ):
			raise ValueError
		if (beta0 <=0 ):
			raise ValueError
		self.alphas = np.ones(K) * alpha0
		self.betas = np.ones(K) * beta0
		self.rewards = np.array([])
		self.armsPlayed = []
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


class UCB:
	def __init__(self, K, c):
		self.ExpectedMeans = np.zeros(K)
		self.NbrPlayArms = np.ones(K)
		self.rewards = np.array([])
		self.armsPlayed = []
		self.c = c
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


class BayesUCB(ThompsonSampling):
	def __init__(self, K, alpha0, beta0, HF, c, horizon):
		super().__init__(K, alpha0, beta0)
		self.HF = HF
		self.c = c
		self.horizon = horizon
	
	def choose_arm(self):
		if self.HF:
		    d = 1/(self.t * np.log(self.t+1)**self.c)
		else:
		    d = 1/(self.t * np.log(self.horizon)**self.c)
		theta = beta.ppf(1 - d , self.alphas, self.betas)
		self.t += 1
		return np.argmax(theta)


class KLUCB(UCB):
	def __init__(self, K, horizon, c, HF):
		super().__init__(K, c)
		self.HF = HF
		self.horizon = horizon

	def SearchingKLUCBIndex(self):
		p = self.ExpectedMeans
		if self.HF:
		    d = (np.log(self.t) + self.c * np.log(np.log(self.t + 1))) / self.NbrPlayArms
		else:
		    d = (np.log(t) + self.c * np.log(np.log(self.horizon))) / self.NbrPlayArms
		return klIC(p,d)

	def choose_arm(self):
			indices = SearchingKLUCBIndex(self)
			self.t += 1
			return np.argmax(indices)


class CPUCB(UCB):
	def __init__(self, K, c):
		super().__init__(K, c)

	def choose_arm(self):
		 _, ic = ssp.proportion_confint(np.floor(self.ExpectedMeans * self.NbrPlayArms), self.NbrPlayArms, 1/ (self.t * np.log(self.t+1)**self.c), method = 'beta')
		self.t += 1
		return np.argmax(ic)

class MOSS(UCB):
	def __init__(self K, c, horizon):
		super().__init__(K, c)
		self.horizon = horizon

	def choose_arm(self):
		K = np.size(self.NbrPlayArms)
		ucb = self.ExpectedMeans + np.sqrt(np.maximum(np.log(self.horizon / (K * self.NbrPlayArms)),0) / self.NbrPlayArms)
		self.t += 1
		return np.argmax(ucb)




