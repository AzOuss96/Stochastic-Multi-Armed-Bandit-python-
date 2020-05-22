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

horizon = 2100 # Overall number of interactions with the environment
dist = 'Bernoulli'
sigma = 0.5

if dist == 'Bernoulli':
	sigma = 0.5

global_switch = True

means = [[0.25, 0.95, 0.2, 0.1, 0.25],
		 [0.3, 0.15, 0.3, 0.85, 0.15],
		 [0.2, 0.1, 0.25, 0.15, 0.9],
		 [0.9, 0.25, 0.15, 0.25, 0.2],
		 [0.35, 0.3, 0.9, 0.3, 0.3]]

periods = [[500, 300, 600, 400, 300]] * len(means)

e = Environment(means,
				periods, 
				horizon=horizon, 
				dist=dist, 
				global_switch=True)

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



iterations = 100
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

MABs = ['AdaptiveThompsonSampling',
		'AdaptiveKLUCB',
		'AdaptiveCPUCB',
		'AdaptiveBayesUCB',
		'Exp3S', 'SWUCB',
		'DiscountedUCB',
		]


'''
#######################################################
					Benchmark
#######################################################
'''

print('Benchmark..')
data = benchmark(MABs, e, iterations, path, **params)

# Saving to CSV
print('Saving to HDF..')
#save(data, 'results_all', path)

'''
#######################################################
				Plots and saving
#######################################################
'''

# Plotting results
print('Plotting figures..')
fig = plt.figure(figsize=(15, 20)) 
gs = gridspec.GridSpec(2, 1, height_ratios=[1, 3])
ax0 = plt.subplot(gs[0])
ax1 = plt.subplot(gs[1])

plot_env(means, periods, fig, ax0)
plot_regrets(data, ax1, fig, path)

fig.savefig(os.path.join(path, 'mabs_all.png'), dpi=100)
plt.show()
