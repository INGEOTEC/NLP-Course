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
from sklearn.cluster import KMeans

X, gold = datasets.make_blobs(random_state=4, 
                              centers=3,
                              n_features=2)
kmeans = KMeans(n_clusters=3, random_state=0).fit(X)
clusters = kmeans.predict(X)

def remove_ticks(ins=None):
    if ins is None:
        ins = plt
    ins.tick_params(axis="both", which="both", bottom=False,
                    top=False, left=False, right=False,
                    labelbottom=False, labelleft=False)

fig, axs = plt.subplots(1, 2)
for cluster, ax, labels in zip([gold, clusters], axs,
                               [[0, 1, 2], [0, 2, 1]]):
    for cl in labels:
        m = cluster == cl
        ax.plot(X[m, 0], X[m, 1], '.')

[(remove_ticks(ax), ax.grid()) for ax in axs]
axs[0].set_title("True Labels")
axs[1].set_title("KMeans")
plt.savefig("kmeans.png", dpi=300)

