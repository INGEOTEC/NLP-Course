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
plt.plot(range(1, len(freq) + 1), freq)
plt.grid()
plt.xlabel('Rank')
plt.ylabel('Frequency')
plt.tight_layout()
plt.savefig('zipf_law.png', dpi=300)


## Log-Log

freq = [f for _, f  in words.most_common()]
plt.loglog(range(1, len(freq) + 1), freq)
plt.grid()
plt.xlabel('Rank')
plt.ylabel('Frequency')
plt.tight_layout()
plt.savefig('zipf_law2.png', dpi=300)
