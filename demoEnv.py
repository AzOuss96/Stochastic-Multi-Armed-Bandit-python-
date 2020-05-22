import sys
sys.path.insert(0, "./")
from env import *
from GLR import *
from MABGLR import *
import ray
from utils import *
from StochasticBanditsPolicies import *
from MABOracle import *
import matplotlib 
from matplotlib import gridspec
sns.set()



horizon = 3000 # Overall number of interactions with the environment
dist = 'Bernoulli'
#means = np.stack([[0.2, 0.8, 0.4],[0.7, 0.1, 0.5],[0.3, 0.5, 0.9]])
global_switch = True
means = [[0.9, 0.2, 0.1], [0.3, 0.8, 0.05], [0.1, 0.3, 0.9]]
periods = [[1000, 1000, 1000], [1000, 1000, 1000], [1000, 1000, 1000]]
env = Environment(means, periods, horizon, global_switch=True)
environment = env.env
switches = env.change_instants
samples = env.generate(dist=dist, sigma=0.05)



sigma = 0.5
alpha0 = 1
beta0 = 1
c = 1
HF = True # Horizon free option for BayesUCB and KLUCB
gamma = 0.8
tau = 100
K = environment.shape[-1]


path = './out/perArmSwitch/'

params = {'alpha0':alpha0, 'beta0':beta0, 'global_switch':global_switch,
		  'c':c, 'K':K, 'HF':HF, 'switches' : switches,
		  'sigma':sigma, 'horizon':horizon,
		  'gamma':gamma, 'tau':tau}

print('Plotting figures..')
#fig, ax = plt.subplots(2, figsize=(15, 20))
fig = plt.figure(figsize=(15, 20)) 
gs = gridspec.GridSpec(3, 1, height_ratios=[1, 2, 1])
ax0 = plt.subplot(gs[0])

plot_env(means, periods, fig, ax0, samples)

plt.show()
