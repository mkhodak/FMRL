import pdb
import h5py
import matplotlib
matplotlib.use('Agg')
import numpy as np
from matplotlib import pyplot as plt
from numpy.linalg import norm
from scipy.optimize import curve_fit


ms = [2**k for k in range(6)]
for name, f, color, marker in [
                               ('Strawman', h5py.File('FMRL/strawman-ftrl-online.h5'), 'maroon', 's'), 
                               ('FMRL', h5py.File('FMRL/fal-ftrl-online.h5'), 'navy', 'v'),
                               ('Omniscient', h5py.File('FMRL/omniscient-ftrl-online.h5'), 'darkorange', 'o'), 
                               ]:
    tar = []
    for m in ms:
        regret = np.array(f[str(m)])[0]
        tar.append(regret.sum() / len(regret))
    plt.plot(ms, tar, label=name, color=color, marker=marker)
    f.close()
plt.xscale('log')
plt.xticks(ms, ms)
plt.xlabel('Samples per Task', fontsize=16)
plt.yscale('log')
plt.ylabel('Log Regret', fontsize=16)
plt.legend(fontsize=14)
plt.savefig('FMRL/strawman.svg')
plt.savefig('FMRL/strawman.png')
plt.clf()

for name, f, color, marker in [
                               ('Single-Task', h5py.File('FMRL/baseline-ogd-online.h5'), 'darkgreen', '^'),
                               ('FLI Variant', h5py.File('FMRL/fli-avg-ogd-online.h5'), 'maroon', 's'),
                               ('FAL Variant', h5py.File('FMRL/fal-ogd-online.h5'), 'navy', 'v'),
                               ('Omniscient', h5py.File('FMRL/omniscient-ogd-online.h5'), 'darkorange', 'o'), 
                               ]:
    tar = []
    for m in ms:
        regret = np.array(f[str(m)])[0]
        tar.append(regret.sum() / len(regret))
    plt.plot(ms, tar, label=name, color=color, marker=marker)
    f.close()
plt.xscale('log')
plt.xticks(ms, ms)
plt.xlabel('Samples per Task', fontsize=16)
plt.yscale('log')
plt.ylabel('Log Regret', fontsize=16)
plt.legend(fontsize=14)
plt.savefig('FMRL/FLI.svg')
plt.savefig('FMRL/FLI.png')
plt.clf()
