# Copyright 2021 Mario Graff Guerrero

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from EvoMSA import base
from microtc.utils import tweet_iterator
from os.path import join, dirname
from collections import Counter
import numpy as np
tweets = join(dirname(base.__file__),
              'tests', 'tweets.json')
T = [x['text'] for x in tweet_iterator(tweets)]
T = [x.strip().lower().split() for x in T]
bow = Counter()
[bow.update(x) for x in T]
words = [w for w, v in bow.items() if v > 1]
w_id = {key: id for id, key in enumerate(words)} 

X = np.zeros((len(T), len(w_id)))
for i, s in enumerate(T):
    index = np.array([j for j in map(w_id.get, s) 
                      if j is not None])
    if index.shape[0]:
        X[i, index] = 1

labels = [x['klass']
          for x in tweet_iterator(tweets)]
labels = np.array(labels)