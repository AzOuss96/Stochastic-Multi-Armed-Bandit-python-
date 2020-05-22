import sys
sys.path.insert(0, "./")
from env import *
from GLR import *
from MABGLR import *
import ray
from utils import *
import seaborn as sns
from StochasticBanditsPolicies import *

sns.set()



horizon = 3000 # Overall number of interactions with the environment
#means = np.stack([[0.2, 0.8, 0.4],[0.7, 0.1, 0.5],[0.3, 0.5, 0.9]])
means = np.stack([[0.2, 0.8], [0.7, 0.1], [0.3, 0.5]])
periods = [[1000, 1000, 1000], [1000, 1000, 1000], [1000, 1000, 1000]]
environment = SwitchBernoulliEnv(means, periods)
sigma = 0.5

alpha0 = 1
beta0 = 1
c = 1
HF = True # Horizon free option for BayesUCB and KLUCB
K = environment.shape[-1]


"""
------------------------------------------------------------------------------
        Thomson sampling in Stationary Bernoulli
-------------------------------------------------------------------------------
"""

TS = ThompsonSampling(K, alpha0, beta0)
TS_armsPlayed = play(TS, environment, horizon)

"""
------------------------------------------------------------------------------
            Thomson sampling with GLR in Stationary Bernoulli
-------------------------------------------------------------------------------
"""


#glr_TS = AdaptiveThompsonSampling(K, alpha0, beta0, sigma)
#GLR_TS_armsPlayed = play(glr_TS, environment, horizon)


"""
------------------------------------------------------------------------------
                              Plotting results
-------------------------------------------------------------------------------
"""
epochs = 1
path = './'
fig, ax = plt.subplots(1, figsize=(15, 10))

print('Thompson Sampling runs..')
TS_data = playruns(TS, environment, horizon, runs, bandit='Thompson Sampling')

print('Thompson Sampling with GLR runs..')
ray.init()
futures = [playGLR_TS.remote(K, alpha0, beta0, sigma, environment, horizon) for _ in range(epochs)]
TS_GLR_data = ray.get(futures)
ray.shutdown()

TS_GLR_data = np.concatenate(TS_GLR_data)

## for plotting
TS_GLR_data = pd.DataFrame(TS_GLR_data, columns=['episod', 'armsPlayed', 'cumulative regret'])
TS_GLR_data['bandit'] = 'Thompson Sampling with GLR'

data = pd.concat((TS_data, TS_GLR_data))
# Saving to CSV
print('Saving to CSV..')
data.to_csv('Thompson_Sampling_GLR'+str(runs)+'.csv', index=False)

sns.lineplot(x='episod', y='cumulative regret', hue='bandit', data=data)
fig.savefig(os.path.join(path, 'Thompson_Sampling_GLR'+str(runs)), dpi=100)
plt.show()

