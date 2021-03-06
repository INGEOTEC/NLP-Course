# Copyright 2021 Mario Graff Guerrero

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
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
[ax.set_title(title) for title, ax in zip(["1st step", "2nd step"], axs)]
plt.savefig("kmeans-2steps.png", dpi=300)

