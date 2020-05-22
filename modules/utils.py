from __future__ import division
import numpy as np
import os
import sys
import ray
import time
from scipy.stats import beta
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import seaborn as sns
import statsmodels.stats.proportion as ssp
import importlib
from scipy.special import kl_div
import tqdm
from tqdm import tqdm

sns.set()

def DivKL(p,q):
    eps = np.spacing(1)
    p = np.maximum(p, eps)
    p = np.minimum(p, 1-eps)
    q = np.maximum(q, eps)
    q = np.minimum(q, 1-eps)
    return  p*np.log(p/q) + (1-p)*np.log((1-p)/(1-q))

def klIC(p,d):
    lM = p.copy()
    uM = np.minimum(1, p + np.sqrt(d/2))
    for j in range(32):
        qM = (uM + lM)/2
        down = kl_div(p, qM) > d
        uM[down] = qM[down]
        lM[down == False] = qM[down == False]
    return uM

def softmax(inputs, scale=1.):
	inputs = inputs - np.max(inputs)  # For numerical stability we use max normalization
	expWeights = np.exp(scale * inputs)
	return expWeights / np.sum(expWeights)

def play(MAB, environment, **kwargs):
	horizon = kwargs['horizon']
	for t in range(horizon):
		armToPlay = MAB.choose_arm()
		reward = PlayBernoulliArm(environment[t, armToPlay])
		MAB.update(reward, armToPlay)
	return np.array(MAB.armsPlayed)	
	
def PlayBernoulliArm(arm):
    return int ((np.random.uniform() < arm) == True)

def to_pandas(data, bandit_name):
	data = np.concatenate(data)
	data = pd.DataFrame(data, columns=['episod', 'armsPlayed', 'cumulative regret'])
	data['bandit'] = bandit_name
	return data

def save(data, file_name, path):
	path = os.path.join(path, file_name)
	data.to_hdf(path, key='results', mode='a')
	#data.to_csv(path, index=False)
	
def name_to_class(class_name):
	for module_ in ['modules.MAB', 'modules.MABGLR', 'modules.MABOracle', 'modules.CascadingMAB']:
		module_ = importlib.import_module(module_)
		try:
			class_ = getattr(module_, class_name)
		except AttributeError as e:
			pass
		else:
			break
	return class_ or None

#def lower_bound_bernoulli(means, switches):

@ray.remote
def play_(bandit_name, environment, **kwargs):
	horizon = kwargs['horizon']
	obs = environment.generate(**kwargs)
	bandit = name_to_class(bandit_name)
	bandit = bandit(**kwargs)
	episods = np.array(range(horizon))
	start = time.time() 
	armsPlayed = play(bandit, obs, **kwargs)
	elapsed = (time.time() - start)
	tqdm.write('elapsed : {}'.format(elapsed))
	data_episod = regret(armsPlayed, obs, horizon)
	return data_episod

def regret(choices, environment, horizon):
	episods = np.array(range(horizon))
	regret = np.max(environment, axis=1) - environment[episods, choices]
	regret = np.cumsum(regret)
	data_episod = np.hstack((episods.reshape(-1,1), choices.reshape(-1,1), regret.reshape(-1,1)))
	return data_episod

def benchmark(MABs, environment, iterations, path, **kwargs):
	all_data = []
	t = tqdm(range(len(MABs)), )
	for i in t:
			bandit_name = MABs[i]
			t.set_description(desc=bandit_name, refresh=True)
			t.refresh()

			# launching workers
			ray.init()
			futures = [play_.remote(bandit_name, environment, **kwargs) for _ in range(iterations)]
			bandit_data = ray.get(futures)
			ray.shutdown()

			# concatenating data 
			bandit_data = to_pandas(bandit_data, bandit_name)
			all_data.append(bandit_data)
			save(bandit_data, bandit_name, path)
	all_data = pd.concat(all_data)
	return all_data
	
def plot_regrets(data, ax, fig, path):
	ax.tick_params(axis="x", labelsize=8)
	ax.tick_params(axis="y", labelsize=8)
	K = data['bandit'].nunique()
	sns.lineplot(x='episod', y='cumulative regret', hue='bandit', data=data, palette = sns.color_palette("husl", K), ax=ax)
	handles, labels = ax.get_legend_handles_labels()
	ax.legend(handles=handles[1:], labels=labels[1:])
	plt.setp(ax.get_legend().get_texts(), fontsize='7') # for legend text


def plot_env(means, num_samples, fig, ax, samples = []):
	ax.tick_params(axis="x", labelsize=8)
	ax.tick_params(axis="y", labelsize=8)
	K = len(num_samples)
	colors = iter(cm.rainbow(np.linspace(0,1,K)))
	for k in range(K):
		c = next(colors)
		# means for arm k
		m = means[k]
		m = np.insert(m, 0, m[0])
		# num samples for arm k
		ns = num_samples[k]
		# change points for arm k
		cp = np.cumsum(ns)
		cp = np.insert(cp, 0, 0.0)

		# plotting samples
		if len(samples) != 0 :
			s = samples[:,k]
			ax.scatter(range(s.size), s,
						marker = 'o',
					 	edgecolors=c, s=9,
						alpha = 0.1, label='arm %d'%k)

		# plotting piecewise means
		ax.step(cp, m, c=c, where='pre', label='arm %d'%k)
		ax.set_xticks(cp, minor=False)
	legend = ['mean arm %d'%k for k in range(K)] + ['obs. arm %d'%k for k in range(K)]
	ax.legend(legend, prop={'size': 6}, loc='upper right')


def plot_decisions(choices, switch_points, ax, fig):
	ax.tick_params(axis="x", labelsize=8)
	ax.tick_params(axis="y", labelsize=8)
	K = np.max(choices) + 1
	horizon = choices.shape[0]
	time = np.arange(horizon)
	colors = iter(cm.rainbow(np.linspace(0, 1, K)))

	isSwitch = np.zeros(horizon)
	for i in range(K):
		isSwitch[switch_points[i]] = 1
	
	for i in range(K):
		c = next(colors)
		t = time[choices == i]
		ax.scatter(t, isSwitch[choices == i], s=9, marker='x', alpha = 0.8, c=[c], label='arm %d'%i)
		legend = ['arm %d'%i for i in range(K)]
	
	for k, v in switch_points.items():
		
		for t in v :
			ax.annotate('%d'%t, (t, 1.), fontsize='xx-small', xytext = (t+10,.95))

	ax.set_ylim((-0.1, 1.1))
	ax.legend(legend, prop={'size': 6})
	

