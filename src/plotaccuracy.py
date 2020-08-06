import pickle
import glob
import sys
from util import *
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
figure(num=None, figsize=(10, 10), dpi=160, facecolor='w', edgecolor='k')

matplotlib.use('tkagg')
font = {'size': 10}
matplotlib.rc('font', **font)
path = './res/'
task = 'acic'
train_data = 100
test_data = 100
paths = []
colors = ['b', 'r', 'y', 'g', 'c']
queries = 1
methods = ['random', 'uncer_y', 'decision_ig']
paths_y = ['-' + name + '-' + str(train_data) + '-' + str(test_data) + '-' +
           str(queries) for name in methods]
models = ['RANDOM', 'UNCER', 'DECISION IG']
y_args = ['acc']
names = ["ENTROPY"]
plot_args = y_args
paths.extend(paths_y)
args = len(plot_args)
for i in range(len(paths)):
    print(paths[i])
    files = glob.glob(path + task + paths[i] + '*')
    files = sorted(files)
    print("filecount")
    print(len(files))
    print(files)
    metric = []
    for fi in files:
        with open(fi, 'rb') as f:
            x = pickle.load(f)
        print("x_query")
        print(x["queryxvals"])
        metric.append(np.array(x['dent']))
    metric = np.array(metric)
    t = np.arange(metric.shape[1])
    res = mean_conf(metric)
    shadedplot(t, res, label=models[i], color=colors[i])
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1),
          fancybox=True, shadow=True, ncol=3)
plt.xlabel('QUERIES')
plt.ylabel('ACCURACY')
#plt.tight_layout()
plt.savefig('./plots/entropy', bbox_inches = "tight")
plt.show()
