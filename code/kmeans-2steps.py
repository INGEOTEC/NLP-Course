from matplotlib import pylab as plt
import numpy as np
from sklearn import datasets
from sklearn.metrics import euclidean_distances
X, _ = datasets.make_blobs(random_state=4,
                           centers=3,
                           n_features=2)
pivots = X[np.array([0, 1, 2])]
clusters = euclidean_distances(X, pivots)
clusters = clusters.argmin(axis=1)
second = np.array([X[cl == clusters].mean(axis=0)
                   for cl in np.unique(clusters)])
second_cl = euclidean_distances(X, second)
second_cl = second_cl.argmin(axis=1)

def remove_ticks(ins=None):
    if ins is None:
        ins = plt
    ins.tick_params(axis="both", which="both", bottom=False,
                    top=False, left=False, right=False,
                    labelbottom=False, labelleft=False)

fig, axs = plt.subplots(1, 2)
for cluster, ax, pivot in zip([clusters, second_cl], axs,
                              [pivots, second]):                 
    for cl in np.unique(cluster):
        m = cluster == cl
        ax.plot(X[m, 0], X[m, 1], '.')
    ax.plot(pivot[:, 0], pivot[:, 1], '*')

[(remove_ticks(ax), ax.grid()) for ax in axs]
plt.savefig("kmeans-2steps.png", dpi=300)

