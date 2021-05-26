from EvoMSA import base
from microtc.utils import tweet_iterator
from os.path import join, dirname
from collections import Counter
tweets = join(dirname(base.__file__),
              'tests', 'tweets.json')
X = [x['text'] for x in tweet_iterator(tweets)]
bow = Counter()
[bow.update(x.strip().lower().split())
 for x in X]
words = [w for w, v in bow.items() if v > 1]
w_id = {key: id for id, key in enumerate(words)} 