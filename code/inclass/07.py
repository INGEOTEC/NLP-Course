from sklearn.svm import LinearSVC
from sklearn.preprocessing import LabelEncoder
import numpy as np
from microtc.utils import tweet_iterator
from b4msa.textmodel import TextModel
from sklearn.metrics import recall_score
from sklearn.model_selection import KFold
from EvoMSA.utils import bootstrap_confidence_interval


D = list(tweet_iterator("semeval2017_En_train.json"))
tm = TextModel(lang="english").fit(D)
# , 
#               token_list=[-1]).fit(D)

le = LabelEncoder().fit([x['klass'] for x in D])
X = tm.transform(D)
y = le.transform([x['klass'] for x in D])

m = LinearSVC().fit(X, y)
recall_score(y, m.predict(X), average=None)

kf = KFold()
hy = np.empty_like(y)
for train, test in kf.split(X):
    m = LinearSVC().fit(X[train], y[train])
    hy[test] = m.predict(X[test])

bootstrap_confidence_interval(y, hy, metric=lambda y, hy: recall_score(y, hy, average=None)[0])


