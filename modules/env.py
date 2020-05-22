from __future__ import division
import numpy as np


class Environment:
	def __init__(self, means, periods, horizon, global_switch=True, dist='Bernoulli', *args, **kwargs):
		self.means = means
		self.periods = periods
		self.global_switch = global_switch
		self.dist = dist
		self.horizon = horizon
		self.K = len(means) ## Number of arms
		self.change_instants = self.change_points(means, periods, horizon)
		self.env = self.env()
		

	def change_points(self, means, periods, horizon):
		switches = [np.cumsum(p[:-1]) for p in periods]
		switches = np.vstack(np.array([np.isin(np.arange(horizon), s) for s in switches])).astype(int) # boolean array of shape (K, horizon) 1 = change and 0 no change
		return switches

	
	def generate(self, *args, **kwargs):
		obs = []
		for arm in range(self.K):
			if self.dist == 'Bernoulli':
				arm_obs = (np.greater(self.env[:,arm], np.random.uniform(0,1,np.size(self.env[:, arm]))) == True) * 1

			elif self.dist == 'Gaussian': 
				sigma = kwargs['sigma']
				arm_obs = np.array([sigma * np.random.randn(self.periods[arm][t]) + self.means[arm][t] for t in range(len(self.means[arm]))])

			else :
				raise ValueError

			obs.append(arm_obs.reshape(-1,1))
		return np.hstack(obs)


	def env(self):
		env = []
		for arm in range(self.K):
			m = self.means[arm]
			p = self.periods[arm]
			arm_env = np.concatenate([m[i] * np.ones((p[i])) for i in range(len(m))]).reshape(-1,1)
			env.append(arm_env)
		return np.hstack(env)
		
