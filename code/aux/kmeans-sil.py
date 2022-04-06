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
from sklearn.metrics import silhouette_score
from sklearn.datasets import load_digits
from sklearn.cluster import KMeans

X, y = load_digits(return_X_y=True)

S = []
for n in range(2, 21):
    kmeans = KMeans(n_clusters=n).fit(X)
    labels = kmeans.predict(X)
    _ = silhouette_score(X, labels)
    S.append(_)

from matplotlib import pylab as plt
def remove_ticks(ins=None):
    if ins is None:
        ins = plt
    ins.tick_params(axis="both", which="both", bottom=False,
                    top=False, left=False, right=False,
                    labelbottom=False, labelleft=False)

plt.plot(range(2, 21), S, '.-')
plt.grid()
plt.xlabel("Number of clusters")
plt.ylabel("Silhouette coefficient")
plt.savefig("kmeans-sil.png", dpi=300)
