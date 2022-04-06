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
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris
from matplotlib import pylab as plt
import numpy as np

X, y = load_iris(return_X_y=True)
pca = PCA(n_components=2).fit(X)
Xn = pca.transform(X)

for cl in np.unique(y):
    m = cl == y
    d = Xn[m]
    plt.plot(d[:, 0], d[:, 1], '.')

plt.grid()
plt.tick_params(axis="both", which="both", bottom=False,
                top=False, left=False, right=False,
                labelbottom=False, labelleft=False)
plt.tight_layout()
plt.savefig("iris-pca.png", dpi=300) 


for x in X[:2]:
    _ = ["$%0.2f$" % i for i in x]
    print("&".join(_) + "\\\\")

hX = pca.inverse_transform(Xn)
for x in hX[:2]:
    _ = ["$%0.2f$" % i for i in x]
    print("&".join(_) + "\\\\")
