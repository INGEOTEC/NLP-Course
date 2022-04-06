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
import numpy as np
from scipy.special import logsumexp
from sklearn.naive_bayes import GaussianNB
X, y = load_iris(return_X_y=True)
classes = np.unique(y)
mu = [X[y == cl].mean(axis=0) for cl in classes]
var = np.array([X[y == cl].var(axis=0)
                for cl in classes])
li = -0.5 * np.sum(np.log(2 * np.pi * var),
                   axis=1)
li = li - 0.5 *\
     np.array([np.sum((X - mu_i)**2 / var_i,axis=1)
               for mu_i, var_i in zip(mu, var)]).T

labels, prior = np.unique(y, return_counts=True)
prior = prior / prior.sum()
hy = li + np.log(prior)
hy = hy - np.atleast_2d(logsumexp(hy, axis=1)).T
hy = np.exp(hy)
hy[100]
# SHOW

m = GaussianNB().fit(X, y)
hhy = m.predict_proba(X)

hhy[100]
# SHOW

cl = m.predict(X)

(y == cl).mean()  #Accuracy

index = np.arange(X.shape[0])
index
# SHOW

np.random.shuffle(index)
index
# SHOW

p = int(index.shape[0] * 0.7)
p
# SHOW
X_train = X[index][:p]
y_train = y[index][:p]
X_test = X[index][p:]
y_test = y[index][p:]

m = GaussianNB().fit(X_train, y_train)
hy = m.predict(X_test)
(y_test == hy).mean() #Accuracy
