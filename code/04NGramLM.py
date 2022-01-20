import numpy as np
from matplotlib import pylab as plt
# %pylab inline

plt.rcParams['text.usetex'] = True
d = 8
l = range(1, 5)
plt.plot(l, [d**i for i in l])
plt.xlabel(r'$\ell$')
plt.ylabel(r'$d^\ell$')
plt.grid()
plt.tight_layout()


# Bigrams 
d = 6
R = np.random.multinomial(1, [1/d] * d, size=10000).argmax(axis=1)
C = np.random.multinomial(1, [1/d] * d, size=10000).argmax(axis=1)
Z = [[2 if c == 1 and np.random.rand() < 0.1 else r, c] for r, c in zip(R, C)]

W = np.zeros((d, d))
for r, c in Z:
    W[r, c] += 1
W = W / W.sum()

for w in (W):
    _ = " & ".join(map(lambda x: "{:0.4f}".format(x), w))
    print(r"{} \\".format(_))
