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
d = 4
R = np.random.multinomial(1, [1/d] * d, size=10000).argmax(axis=1)
C = np.random.multinomial(1, [1/d] * d, size=10000).argmax(axis=1)
rand = np.random.rand
Z = [[r, 2 if r == 1 and rand() < 0.1 else c]
      for r, c in zip(R, C)
     if r != c or (r == c and rand() < 0.2)]

W = np.zeros((d, d))
for r, c in Z:
    W[r, c] += 1
W = W / W.sum()

for w in W:
    _ = " & ".join(map(lambda x: "{:0.4f}".format(x), w))
    print(r"{} \\".format(_))

M_r = W.sum(axis=1)
p_l = (W / np.atleast_2d(M_r).T)

for w in p_l:
    _ = " & ".join(map(lambda x: "{:0.4f}".format(x), w))
    print(r"{} \\".format(_))
