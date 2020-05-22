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

horizon = 2000 # Overall number of interactions with the environment
dist = 'Bernoulli'
sigma = 0.05
global_switch = False
means = [[0.1, 0.8], [0.2, 0.95, 0.7]]
periods = [[700, 1300], [500, 1000, 500]]

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



iterations = 10
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
		'AdaptiveBayesUCB',
		'Exp3S', 'SWUCB',
		'DiscountedUCB',
		'OracleThompsonSampling',
		'OracleKLUCB',
		'OracleBayesUCB'
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
