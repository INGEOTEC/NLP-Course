import numpy as np
from b4msa.textmodel import TextModel
from EvoMSA.tests.test_base import TWEETS
from microtc.utils import tweet_iterator
from sklearn.model_selection import StratifiedKFold
from EvoMSA.utils import LabelEncoderWrapper, bootstrap_confidence_interval
from EvoMSA.model import Multinomial
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

D = [(x['text'], x['klass']) for x in tweet_iterator(TWEETS)]
y = [y for _, y in D]
le = LabelEncoderWrapper().fit(y)
y = le.transform(y)

tm = TextModel(token_list=[-1], 
               weighting='tf').fit([x for x, _ in D])

folds = StratifiedKFold(shuffle=True, random_state=0)

hy = np.empty(len(D))
for tr, val in folds.split(D, y):
    _ = [D[x][0] for x in tr]
    X = tm.transform(_)
    m = LogisticRegression(multi_class='multinomial').fit(X, y[tr])
    # m = LinearSVC().fit(X, y[tr])
    _ = [D[x][0] for x in val]
    hy[val] = m.predict(tm.transform(_))

ci = bootstrap_confidence_interval(y, hy)
ci
(0.2839760475399691, 0.30881116416736665)

tm = TextModel(token_list=[-1]).fit([x for x, _ in D])
hy = np.empty(len(D))
for tr, val in folds.split(D, y):
    _ = [D[x][0] for x in tr]
    X = tm.transform(_)
    m = LogisticRegression(multi_class='multinomial').fit(X, y[tr])
    # m = LinearSVC().fit(X, y[tr])
    _ = [D[x][0] for x in val]
    hy[val] = m.predict(tm.transform(_))

ci = bootstrap_confidence_interval(y, hy)
ci

