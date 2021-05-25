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
