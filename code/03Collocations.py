from numpy.core.fromnumeric import size
from text_models import Vocabulary
from collections import Counter
import numpy as np

# Collocations

date = dict(year=2022, month=1, day=10)
voc = Vocabulary(date, lang='En')
bigrams = Counter({k: v for k, v in voc.voc.items() if k.count("~")})
co_occurrence = np.zeros((5, 5))
index = {'the': 0, 'to': 1, 'of': 2, 'in': 3, 'and': 4}

for bigram, cnt in bigrams.most_common():
    a, b = bigram.split('~')
    if a in index and b in index:
        co_occurrence[index[a], index[b]] = cnt
        co_occurrence[index[b], index[a]] = cnt

keys = list(index.items())
keys.sort(key=lambda x: x[1])

print(' | ' + ' | '.join([k for k, _ in keys]) + ' | ')
for c, (k, _) in zip(co_occurrence, keys):
    _ = " | ".join(map(lambda x: '{: 7d}'.format(int(x)), c))
    print('{} | {} |'.format(k, _))


for bigram, cnt in bigrams.items():
    a, b = bigram.split('~')
    for x in [a, b]:
        if x not in index:
            index[x] = len(index)
len(index)

# Bernoulli Distribution
x = np.random.binomial(1, 0.3, size=1000)
hp = x.mean()

# Categorical distribution
X = np.random.multinomial(1, [1/6] * 6, size=100)
x = X.argmax(axis=1)

var, counts = np.unique(x, return_counts=True)
N = counts.sum()
p = counts / N