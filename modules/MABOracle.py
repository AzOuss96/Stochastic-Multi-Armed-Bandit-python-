import numpy as np
from modules.MAB import *

## TO DO switch to metaclass / decorator implementation 

class OracleThompsonSampling(ThompsonSampling):
	def __init__(self, K, alpha0, beta0, switches, *args, **kwargs):
		super().__init__(K, alpha0, beta0, *args, **kwargs)
		self.switches = switches  ## boolean array of shape = (K, N) 
	
	def update(self, reward, armToPlay):
		super().update(reward, armToPlay)
		if self.t in self.switches[armToPlay]: 
			super().reset(hist=False, arm=armToPlay)
		

class OracleKLUCB(KLUCB):
	def __init__(self, K, horizon, c, HF, switches, *args, **kwargs):
		super().__init__(K, horizon, c, HF, *args, **kwargs)
		self.switches = switches
	
	def update(self, reward, armToPlay):
		super().update(reward, armToPlay)
		if self.t in self.switches[armToPlay]: 
			super().reset(hist=False, arm=armToPlay)
		

class OracleBayesUCB(BayesUCB):
	def __init__(self, K, alpha0, beta0, HF, c, horizon, switches, *args, **kwargs):
		super().__init__(K, alpha0, beta0, HF, c, horizon, *args, **kwargs)
		self.switches = switches
	
	def update(self, reward, armToPlay):
		super().update(reward, armToPlay)
		if self.t in self.switches[armToPlay]: 
			super().reset(hist=False, arm=armToPlay)
		


