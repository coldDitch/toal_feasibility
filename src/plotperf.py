import pickle
import glob
import sys
from util import *
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
figure(num=None, figsize=(10, 6), dpi=160, facecolor='w', edgecolor='k')

matplotlib.use('tkagg')
font = {'size': 10}
matplotlib.rc('font', **font)
path = './res/'
task = 'multilin'
train_data = 10
test_data = 30
paths = []
colors = ['b', 'r', 'y', 'g', 'c']
queries = 1
methods = ['random', 'uncer_y', 'decision_ig']
paths_y = ['-' + name + '-' + str(train_data) + '-' + str(test_data) + '-' +
           str(queries) for name in methods]
models = ['RANDOM', 'UNCER', 'DECISION IG']
y_args = ['acc']
names = ["ACCURACY"]
plot_args = y_args
paths.extend(paths_y)
args = len(plot_args)

for i in range(len(paths)):
    for j in range(args):
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
            metric.append(np.array(x[plot_args[j]]))
        metric = np.array(metric)
        t = np.arange(metric.shape[1])
        res = bootstrap_results(metric)
        plt.subplot(3, max((1,round(args / 2))), j + 1)
        plt.title(names[j])
        plt.plot(t, np.mean(metric, axis=0), color=colors[i])
        shadedplot(t, res, label=paths[i], color=colors[i])
plt.subplot(3,max((1,round(args / 2))),3*max((1,round(args / 2))))
for i in range(len(paths)):
    plt.plot([1],label=models[i], color=colors[i])
plt.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    left=False,
    right=False,
    labelbottom=False) # labels along the bottom edge are off§
plt.legend()
plt.tight_layout()
plt.savefig('./plots/perfplot')
plt.show()
