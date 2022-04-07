import numpy as np
from b4msa.textmodel import TextModel
from EvoMSA.tests.test_base import TWEETS
from microtc.utils import tweet_iterator
from sklearn.model_selection import StratifiedKFold
from EvoMSA.utils import LabelEncoderWrapper, bootstrap_confidence_interval
from sklearn.linear_model import LogisticRegression


D = [(x['text'], x['klass']) for x in tweet_iterator(TWEETS)]
y = [y for _, y in D]
le = LabelEncoderWrapper().fit(y)
y = le.transform(y)

tm = TextModel(token_list=[-1], weighting='tfidf').fit([x for x, _ in D])

folds = StratifiedKFold(shuffle=True, random_state=0)

hy = np.empty(len(D))
for tr, val in folds.split(D, y):
    _ = [D[x][0] for x in tr]
    X = tm.transform(_)
    m = LogisticRegression().fit(X, y[tr])
    _ = [D[x][0] for x in val]
    hy[val] = m.predict(tm.transform(_))

ci = bootstrap_confidence_interval(y, hy)
ci