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
