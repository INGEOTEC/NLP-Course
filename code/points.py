from matplotlib import pylab as plt
import numpy as np
from sklearn import datasets
from sklearn.datasets import load_iris


def remove_ticks(ins=None):
    if ins is None:
        ins = plt
    ins.tick_params(axis="both", which="both", bottom=False,
                    top=False, left=False, right=False,
                    labelbottom=False, labelleft=False)


X, _ = datasets.make_blobs(random_state=4, centers=3, n_features=3)
remove_ticks()
plt.grid()
plt.plot(X[:, 0], X[:, 1], '.')
plt.tight_layout()
plt.savefig("points-01.png", dpi=300)


for x in X[:, np.array([0, 1])]:
    _ = ["$%0.2f$" % i for i in x]
    print("&".join(_) + "\\\\")


for x in X:
    _ = ["$%0.2f$" % i for i in x]
    print("&".join(_) + "\\\\")


fig, (ax1, ax2, ax3) = plt.subplots(3)
ax1.plot(X[:, 0], X[:, 1], '.')
ax2.plot(X[:, 0], X[:, 2], '.')
ax3.plot(X[:, 1], X[:, 2], '.')
ax1.set_title("Ch. 1 / Ch. 2")
ax2.set_title("Ch. 1 / Ch. 3")
ax3.set_title("Ch. 2 / Ch. 3")
[ins.grid() for ins in [ax1, ax2, ax3]]
[remove_ticks(ins) for ins in [ax1, ax2, ax3]]
plt.tight_layout()
plt.savefig("points-01-02-12.png", dpi=300) 


X, y = load_iris(return_X_y=True)
for cl, name in zip(np.unique(y),
                    ['setosa', 'versicolor', 'virginica']):
    m = y == cl
    _ = ["$%0.2f$" % i for i in X[m][0]] + [name]
    print("&".join(_) + "\\\\")


fig, (ax1, ax2) = plt.subplots(1, 2)
for cl in np.unique(y):
    m = y == cl
    ax1.plot(X[m, 0], X[m, 1], '.')
    ax2.plot(X[m, 0], X[m, 2], '.')
ax1.set_title("Sepal Length / Sepal Width")
ax2.set_title("Sepal Length / Petal Length")
[ins.grid() for ins in [ax1, ax2]]
[remove_ticks(ins) for ins in [ax1, ax2]]
plt.tight_layout()
plt.savefig("iris.png", dpi=300) 

