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
