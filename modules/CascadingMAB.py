from MAB import SWUCB, KLUCB, DUCB


class CascadingSWUCB(SWUCB):
	def __init__(self, K, c, tau, ntop, *args, **kwargs):
		super(CascadingSWUCB, self).__init__(K, c, *args, **kwargs)
		assert tau >= 1, "ValueError: tau has to be superior or equal to 1."
		self.tau = tau
		self.lastRewards = np.zeros(tau) 
		self.lastArmsPlayed = np.full(tau, -1) 
		self.ntop = ntop ## n best items to recommend
	
	def top_utilities(self):
		evals = [self.evaluate(arm) for arm in range(self.K)]
		




