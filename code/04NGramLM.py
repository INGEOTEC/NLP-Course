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
_ = map(lambda x: "{:4f}".format(x), W.sum(axis=0))
print(", ".join(_))
p_l = (W / np.atleast_2d(M_r).T)

for w in p_l:
    _ = " & ".join(map(lambda x: "{:0.4f}".format(x), w))
    print(r"{} \\".format(_))

## Generating sequences

cat = lambda x: np.random.multinomial(1, x, 1).argmax()
id2word = {0: 'a', 1: 'b', 2: 'c', 3: 'd'}
w1 = cat(M_r)

text = [cat(M_r)]
l = 25
while len(text) < l:
    next = cat(p_l[text[-1]])
    text.append(next)
text = " ".join(map(lambda x: id2word[x], text))
text


w2id = {v: k for k, v in id2word.items()}
lst = [w2id[x] for x in text.split()]
Z = [[a, b] for a, b in zip(lst, lst[1:])]

d = len(w2id)
W = np.zeros((d, d))
for r, c in Z:
    W[r, c] += 1
W = W / W.sum()

for w in W:
    _ = " & ".join(map(lambda x: "{:0.4f}".format(x), w))
    print(r"{} \\".format(_))


text = 'a d b c'
lst = [w2id[x] for x in text.split()]
p = M_r[lst[0]]
for a, b in zip(lst, lst[1:]):
    p *= p_l[a, b]
