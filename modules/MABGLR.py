import numpy as np
from modules.GLR import *
from modules.MAB import *

## TO DO switch to metaclass / decorator implementation 

class AdaptiveThompsonSampling(ThompsonSampling):
	def __init__(self, K, alpha0, beta0, sigma, global_switch = True , *args, **kwargs):
		super().__init__(K, alpha0, beta0, *args, **kwargs)
		self.global_switch = global_switch
		self.changes = dict([(i, []) for i in range(K)])
		self.change_detectors = [ImprovedGLR(sigma) for _ in range(K)]
		self.last_change = 0
	
	def update(self, reward, armToPlay):
		super().update(reward, armToPlay)	
		restart = self.change_detectors[armToPlay].process(reward)
		if restart == 1: 
			self.changes[armToPlay].append(self.t + self.last_change)
			self.last_change = self.t
			if self.global_switch == True :
				for i in range(self.K):  ## restarting GLR
					self.change_detectors[i].restarting()
				super().reset(hist=False, arm=-1)
			else :
				self.change_detectors[armToPlay].restarting()
				super().reset(hist=False, arm=armToPlay)


class AdaptiveKLUCB(KLUCB):
	def __init__(self, K, horizon, c, HF, sigma, global_switch = True, *args, **kwargs):
		super().__init__(K, horizon, c, HF, *args, **kwargs)
		self.global_switch = global_switch
		self.change_detectors = [ImprovedGLR(sigma) for _ in range(K)]
		self.changes = dict([(i, []) for i in range(K)])
		self.last_change = 0

	def update(self, reward, armToPlay):
		super().update(reward, armToPlay)	
		restart = self.change_detectors[armToPlay].process(reward)
		if restart == 1: 
			self.changes[armToPlay].append(self.t + self.last_change)
			self.last_change = self.t
			if self.global_switch == True :
				for i in range(self.K):
					self.change_detectors[i].restarting()
				super().reset(hist=False, arm=-1)
			else :
				self.change_detectors[armToPlay].restarting()
				super().reset(hist=False, arm=armToPlay)


class AdaptiveBayesUCB(BayesUCB):
	def __init__(self, K, alpha0, beta0, HF, c, horizon, sigma, global_switch = True, *args, **kwargs):
		super().__init__(K, alpha0, beta0, HF, c, horizon, *args, **kwargs)
		self.global_switch = global_switch
		self.change_detectors = [ImprovedGLR(sigma) for _ in range(K)]
		self.changes = dict([(i, []) for i in range(K)])
		self.last_change = 0

	def update(self, reward, armToPlay):
		super().update(reward, armToPlay)	
		restart = self.change_detectors[armToPlay].process(reward)
		if restart == 1: 
			self.changes[armToPlay].append(self.t + self.last_change)
			self.last_change = self.t
			if self.global_switch == True :
				for i in range(self.K):
					self.change_detectors[i].restarting()
				super().reset(hist=False, arm=-1)
			else :
				self.change_detectors[armToPlay].restarting()
				super().reset(hist=False, arm=armToPlay)



class AdaptiveCPUCB(CPUCB):
	def __init__(self, K, c, sigma, global_switch = True, *args, **kwargs):
		super(AdaptiveCPUCB, self).__init__(K, c, *args, **kwargs)
		self.global_switch = global_switch
		self.change_detectors = [ImprovedGLR(sigma) for _ in range(K)]
		self.changes = dict([(i, []) for i in range(K)])
		self.last_change = 0

	def update(self, reward, armToPlay):
		super().update(reward, armToPlay)	
		restart = self.change_detectors[armToPlay].process(reward)
		if restart == 1: 
			self.changes[armToPlay].append(self.t + self.last_change)
			self.last_change = self.t
			if self.global_switch == True :
				for i in range(self.K):
					self.change_detectors[i].restarting()
				super().reset(hist=False, arm=-1)
			else :
				self.change_detectors[armToPlay].restarting()
				super().reset(hist=False, arm=armToPlay)

