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



