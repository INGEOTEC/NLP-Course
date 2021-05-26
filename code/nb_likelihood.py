from sklearn.datasets import load_iris
import numpy as np
X, y = load_iris(return_X_y=True)
classes = np.unique(y)
mu = [X[y == cl].mean(axis=0) for cl in classes]
var = np.array([X[y == cl].var(axis=0)
                for cl in classes])
li = -0.5 * np.sum(np.log(2 * np.pi * var),
                   axis=1)
li = li - 0.5 *\
     np.array([np.sum((X - mu_i)**2 / var_i, axis=1)
               for mu_i, var_i in zip(mu, var)]).T