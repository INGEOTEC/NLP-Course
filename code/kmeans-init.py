from matplotlib import pylab as plt
import numpy as np
from sklearn import datasets
from sklearn.datasets import load_iris
from sklearn.metrics import euclidean_distances


def remove_ticks(ins=None):
    if ins is None:
        ins = plt
    ins.tick_params(axis="both", which="both", bottom=False,
                    top=False, left=False, right=False,
                    labelbottom=False, labelleft=False)


X, _ = datasets.make_blobs(random_state=4, centers=3,
                           n_features=2)


fig, (ax1, ax2) = plt.subplots(1, 2)
for pivots, ax in zip([[0, 1, 2], [4, 6, 0]], [ax1, ax2]):
    pivots = X[pivots]                            
    clusters = euclidean_distances(X, pivots).argmin(axis=1)
    for cl in np.unique(clusters):
        m = clusters == cl
        ax.plot(X[m, 0], X[m, 1], '.')
    ax.plot(pivots[:, 0], pivots[:, 1], '*')

[(remove_ticks(ax), ax.grid()) for ax in [ax1, ax2]]
plt.tight_layout()
plt.savefig("k-means-init.png", dpi=300)