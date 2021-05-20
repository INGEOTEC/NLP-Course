from matplotlib import pylab as plt
import numpy as np
from sklearn import datasets
plt.tick_params(axis="both", which="both", bottom=False,
                top=False, left=False, right=False,
                labelbottom=False, labelleft=False)

plt.grid()
X, _ = datasets.make_blobs(random_state=1, centers=3)
plt.tight_layout()
plt.plot(X[:, 0], X[:, 1], '.')
plt.tight_layout()
plt.savefig("points.png")