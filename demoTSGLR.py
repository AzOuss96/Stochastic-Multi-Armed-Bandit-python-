import sys
sys.path.insert(0, "./")
from modules.env import *
from modules.GLR import *
from modules.MABGLR import *
import ray
from modules.utils import *
from modules.MAB import *
from modules.MABOracle import *
import matplotlib 
from matplotlib import gridspec
sys.path.append('./modules')


'''
#######################################################
		  		Environment creation
#######################################################
'''

horizon = 3000 # Overall number of interactions with the environment
global_switch = True

dist = 'Gaussian'
sigma = 0.5

if dist == 'Bernoulli':
	sigma = 0.5


means = [[0.9, 0.2, 0.1], [0.3, 0.8, 0.05], [0.1, 0.3, 0.9]]
periods = [[1000, 1000, 1000], [1000, 1000, 1000], [1000, 1000, 1000]]


e = Environment(means,
				periods, 
				horizon=horizon, 
				dist=dist, 
				global_switch=global_switch)

environment = e.env
switches = e.change_instants
samples = e.generate(sigma=sigma)


'''
#######################################################
					MAB parameters
#######################################################
''' 

alpha0 = 1
beta0 = 1
c = 1
HF = True # Horizon free option for BayesUCB and KLUCB
gamma = 0.8
tau = 100
K = environment.shape[-1]

path = './out/perArmSwitch/'

params = {'alpha0':alpha0, 'beta0':beta0, 
		  'global_switch':global_switch,
		  'c':c, 'K':K, 'HF':HF, 
		  'switches' : switches,
		  'sigma':sigma, 'horizon':horizon,
		  'gamma':gamma, 'tau':tau}


'''
#######################################################
					MAB policies
#######################################################
'''

TSGLR = AdaptiveThompsonSampling(**params)
bandit_name = 'AdaptiveThompsonSampling'


'''
#######################################################
					Benchmark
#######################################################
'''
print('Benchmark..')
arms_played = play(TSGLR, environment, **params)
regret = regret(arms_played, environment, horizon)
regret = to_pandas([regret], bandit_name)
switch_points = TSGLR.changes

print('Detected change points :')
for k, v in switch_points.items():
	if len(v) == 0:
		pass
	else:
		for t in v:
			print('Arm {}, instant {}'.format(k, t))

'''
#######################################################
				Plots and saving
#######################################################
'''
print('Plotting figures..')
fig = plt.figure(figsize=(15, 20)) 
gs = gridspec.GridSpec(3, 1, height_ratios=[1, 2, 1])
ax0 = plt.subplot(gs[0])
ax1 = plt.subplot(gs[1])
ax2 = plt.subplot(gs[2])

plot_env(means, periods, fig, ax0, samples=samples)
plot_regrets(regret, ax1, fig, path)
plot_decisions(arms_played, switch_points, ax2, fig)

fig.savefig(os.path.join(path, 'regrets.png'), dpi=100)
plt.show()
