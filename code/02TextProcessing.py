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
coef = np.linalg.lstsq(X, freq, rcond=None)[0]
coef
hy = np.dot(X, coef)
plt.plot(rank, freq, '.')
plt.plot(rank, hy)
plt.legend(["Measured", "Predicted"])
plt.grid()
plt.xlabel('Inverse Rank')
plt.ylabel('Frequency')
plt.tight_layout()
plt.savefig('zipf_law3.png', dpi=300)

