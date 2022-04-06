from sklearn.metrics import recall_score
from sklearn.model_selection import KFold
from EvoMSA import base
from microtc.utils import tweet_iterator
from sklearn.preprocessing import LabelEncoder
import numpy as np


D = list(tweet_iterator("semeval2017_En_train.json"))
y = np.array([x['klass'] for x in D])

kf = KFold(shuffle=True)
hy = np.empty_like(y)
for train, test in kf.split(D):
    m = base.EvoMSA(Emo=True,
                    stacked_method="sklearn.naive_bayes.GaussianNB",
                    lang="en").fit([D[x] for x in train], 
                                   [D[x]['klass'] for x in train])
    hy[test] = m.predict([D[x] for x in test])

recall_score(y, hy, average=None)