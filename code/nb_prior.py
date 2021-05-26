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
from sklearn.datasets import load_iris
from scipy.special import logsumexp
from sklearn.naive_bayes import GaussianNB
import numpy as np
X, y = load_iris(return_X_y=True)
labels, prior = np.unique(y, return_counts=True)
prior = prior / prior.sum()
hy = li + np.log(prior)
hy = hy - np.atleast_2d(logsumexp(hy, axis=1)).T
hy = np.exp(hy)

m = GaussianNB().fit(X, y)
hhy = m.predict_proba(X)
