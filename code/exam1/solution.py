cdn = ["a", "b", "c", "d"]
a = {k: v for k, v in enumerate(cdn)}
print(a)

cdn = ["a", "b", "c", "d"]
a = {v: k for k, v in enumerate(cdn)}
print(a)

lst = [i for i in range(10) if i % 2]
print(lst)

from collections import Counter

cnt = {('a', 'z'): 3, ('a', 'b'): 1, 
       ('z', 'a'): 1, ('b', 'z'): 2}
output = Counter()
for k, v in cnt.items():
    # key = k[:-1]
    key = k[1:]
    output.update({key: v})
print(output)

import numpy as np
W = np.array([[1, 2, 3], [1, 1, 1]])
print(W.sum(axis=1))


def func(*args):
    import numpy as np
    output = 0
    for a in args:
        output += np.log(a)
    return np.exp(output)


from matplotlib import pylab as plt
def num_tokens_voc_size(fname):
    from microtc.utils import tweet_iterator
    lst = [x['text'] for x in tweet_iterator(fname)]
    pos = list(range(0, len(lst), 500)) + [-1]
    tot = Counter()
    ntokens_vocs = []
    for start, end in zip(pos, pos[1:]):
        for x in lst[start:end]:
            words = x.split()
            tot.update(words)
        ntokens_vocs.append([sum(list(tot.values())), len(tot)])
    return ntokens_vocs

output = num_tokens_voc_size('tweets-2022-02-13.json.gz')
output[:3]
plt.plot([x for x, y in output], [y for x, y in output])
plt.xlabel('Number of tokens')
plt.ylabel('Vocabulary size')
plt.grid()
plt.tight_layout()
plt.savefig('language_law.png', dpi=300)


from scipy.optimize import minimize
def f(args):
    c1, beta, c2 = args
    x = output[:, 0]
    y = output[:, 1]
    return ((y - c1 * x**beta + c2)**2).sum()

output = np.array(output)
res = minimize(f, np.array([0.5, 0.5, 0.5]))
res.x

c1, beta, c2 = res.x
plt.plot([x for x, y in output], [y for x, y in output])
x = output[:, 0]
plt.plot(output[:, 0], c1 * x**beta + c2)
