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
from EvoMSA.model import Bernoulli
from EvoMSA.utils import LabelEncoder, bootstrap_confidence_interval
from microtc.textmodel import TextModel
from microtc.utils import tweet_iterator
from os.path import join, dirname
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.model_selection import StratifiedKFold

tweets = join(dirname(base.__file__),
              'tests', 'tweets.json')
D = list(tweet_iterator(tweets))
y = [x['klass'] for x in D]
le = LabelEncoder().fit(y)
y = le.transform(y)

tm = TextModel(token_list=[-1]).fit(D)
X = tm.transform(D)
m = Bernoulli().fit(X, y)
print((y == m.predict(X)).mean())
# 0.724

_ = train_test_split(D, y, test_size=0.2)
Xtrain, Xtest, ytrain, ytest = _
tm = TextModel(token_list=[-1]).fit(Xtrain)
m = Bernoulli().fit(tm.transform(Xtrain), ytrain)
hy = m.predict(tm.transform(Xtest))
print((ytest == hy).mean())
# 0.55

folds = StratifiedKFold(n_splits=5,
                        shuffle=True, random_state=0)
hy = np.empty_like(y)
for tr, ts in folds.split(D, y):
    _ = [D[x] for x in tr]
    tm = TextModel(token_list=[-1]).fit(_)
    m = Bernoulli().fit(tm.transform(_), y[tr])
    _ = [D[x] for x in ts]
    hy[ts] = m.predict(tm.transform(_))
bootstrap_confidence_interval(y, hy, 
            metric=lambda y, hy: (y == hy).mean())
# (0.522, 0.575)