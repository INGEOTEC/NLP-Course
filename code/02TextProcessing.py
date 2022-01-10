# Introduction

## Frequency of words
from microtc.utils import tweet_iterator
from EvoMSA.tests.test_base import TWEETS

words = dict()
for tw in tweet_iterator(TWEETS):
    text = tw['text']
    for w in text.split():
        key = w.strip()
        try:
            words[key] += 1
        except KeyError:
            words[key] = 1

words['si']

## Simpler version using Counter

from microtc.utils import tweet_iterator
from EvoMSA.tests.test_base import TWEETS
from collections import Counter

words = Counter()
for tw in tweet_iterator(TWEETS):
    text = tw['text']
    words.update([x.strip() for x in text.split()])

words['si']

# Zipf's Law

%pylab inline
from matplotlib import pylab as plt

freq = [f for _, f  in words.most_common()]
rank = range(1, len(freq) + 1)
plt.plot(rank, freq, '.')
plt.grid()
plt.xlabel('Rank')
plt.ylabel('Frequency')
plt.tight_layout()
plt.savefig('zipf_law.png', dpi=300)


## Inverse rank
import numpy as np

freq = [f for _, f  in words.most_common()]
rank = 1 / np.arange(1, len(freq) + 1)
plt.plot(rank, freq, '.')
plt.grid()
plt.xlabel('Inverse Rank')
plt.ylabel('Frequency')
plt.tight_layout()
plt.savefig('zipf_law2.png', dpi=300)

## OLS

X = np.atleast_2d(rank).T
c = np.linalg.lstsq(X, freq, rcond=None)[0]
c
hy = np.dot(X, c)
plt.plot(rank, freq, '.')
plt.plot(rank, hy)
plt.legend(['Measured', 'Predicted'])
plt.grid()
plt.xlabel('Inverse Rank')
plt.ylabel('Frequency')
plt.tight_layout()
plt.savefig('zipf_law3.png', dpi=300)

# Heaps' Law

words = Counter()
tokens_voc= list()
for tw in tweet_iterator(TWEETS):
    text = tw['text']
    words.update([x.strip() for x in text.split()])
    tokens_voc.append([sum(list(words.values())),
                       len(words)])

n = [x[0] for x in tokens_voc]
v = [x[1] for x in tokens_voc]
plt.plot(n, v, '.')
plt.grid()
plt.xlabel('Number of tokens')
plt.ylabel('Vocabulary Size')
plt.tight_layout()
plt.savefig('heaps_law.png', dpi=300)

## Optimization

from scipy.optimize import minimize
n = np.array(n)
v = np.array(v)
def f(w):
    k, beta = w
    return ((v - k * n**beta)**2).sum()

res = minimize(f, np.array([1, 0.5]))
k, beta = res.x
k, beta

plt.plot(n, v, '.')
plt.plot(n, k*n**beta)
plt.legend(['Measured', 'Predicted'])
plt.grid()
plt.xlabel('Number of tokens')
plt.ylabel('Vocabulary Size')
plt.tight_layout()
plt.savefig('heaps_law2.png', dpi=300)
