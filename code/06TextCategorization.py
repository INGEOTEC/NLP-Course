from typing import Counter
from matplotlib.pyplot import axis
import numpy as np
from b4msa.textmodel import TextModel
from EvoMSA.tests.test_base import TWEETS
from microtc.utils import tweet_iterator
from scipy.stats import norm, multinomial
from matplotlib import pylab as plt
from collections import Counter
from sklearn.model_selection import StratifiedKFold
from scipy.special import logsumexp
# %pylab inline

plt.rcParams['text.usetex'] = True

pos = norm(loc=3, scale=2.5)
neg = norm(loc=-0.5, scale=0.75)
plt.clf()
for ins, color in zip([pos, neg], ['b', 'r']):
    x = np.linspace(ins.ppf(0.001),
                    ins.ppf(0.999))
    plt.plot(x, ins.pdf(x), color)
plt.legend(['Positive', 'Negative'])
plt.grid()
plt.tight_layout()
plt.savefig('two_normals.png', dpi=300)

_min = neg.ppf(0.05)
_max = neg.ppf(0.95)
D = [(x, 0) for x in neg.rvs(100) if x >= _min and x <= _max]
D += [(x, 1) for x in pos.rvs(1000) if x < _min or x > _max]

plt.clf()
plt.plot([x for x, k in D if k==1],
         pos.pdf([x for x, k in D if k==1]), 'b.')
plt.plot([x for x, k in D if k==0],
         neg.pdf([x for x, k in D if k==0]), 'r.')
plt.grid()
plt.legend(['Positive', 'Negative'])
plt.tight_layout()
plt.savefig('two_normal_samples.png', dpi=300)

l_pos = norm(loc=np.mean([x for x, k in D if k == 1]),
             scale=np.std([x for x, k in D if k == 1]))
l_neg = norm(loc=np.mean([x for x, k in D if k == 0]),
             scale=np.std([x for x, k in D if k == 0]))            

_, priors = np.unique([k for _, k in D], return_counts=True)
N = priors.sum()
prior_pos = priors[1] / N
prior_neg = priors[0] / N

x = np.array([x for x, _ in D])
x.sort()
post_pos = l_pos.pdf(x) * prior_pos
post_neg = l_neg.pdf(x) * prior_neg

post = np.vstack([post_pos, post_neg])
evidence = post.sum(axis=0)
post_pos /= evidence
post_neg /= evidence

plt.clf()
plt.plot(x, post_pos, 'b')
plt.plot(x, post_neg, 'r')

plt.legend([r'$P(\mathcal Y=1 \mid \mathcal X)$',
            r'$P(\mathcal Y=0 \mid \mathcal X)$'])
plt.tight_layout()
plt.grid()
plt.savefig('two_classes_posteriori.png', dpi=300)


klass = lambda x: 1 if l_pos.pdf(x) * prior_pos > l_neg.pdf(x) * prior_neg else 0

plt.clf()
plt.plot([x for x, k in D if k==1 and klass(x) == 1],
         pos.pdf([x for x, k in D if k==1 and klass(x) == 1]), 'k.')
plt.plot([x for x, k in D if k==0 and klass(x) == 0],
         neg.pdf([x for x, k in D if k==0 and klass(x) == 0]), 'k.')

plt.plot([x for x, k in D if k==1 and klass(x) == 0],
         pos.pdf([x for x, k in D if k==1 and klass(x) == 0]), 'r.')
plt.plot([x for x, k in D if k==0 and klass(x) == 1],
         neg.pdf([x for x, k in D if k==0 and klass(x) == 1]), 'r.')

plt.tight_layout()
plt.grid()
plt.savefig('two_classes_posteriori_error.png', dpi=300)

# Categorical distribution 

m = {k: chr(122 - k) for k in range(4)}
pos = multinomial(1, [0.20, 0.20, 0.35, 0.25])
neg = multinomial(1, [0.35, 0.20, 0.25, 0.20])
length = norm(loc=10, scale=3)
D = []
id2w = lambda x: " ".join([m[_] for _ in x.argmax(axis=1)])
for l in length.rvs(size=1000):
    D.append((id2w(pos.rvs(round(l))), 1))
    D.append((id2w(neg.rvs(round(l))), 0))

D[:5]

D_pos = []
[D_pos.extend(data.split()) for data, k in D if k == 1]
D_neg = []
[D_neg.extend(data.split()) for data, k in D if k == 0]

words, l_pos = np.unique(D_pos, return_counts=True)
l_pos = l_pos / l_pos.sum()
_, l_neg = np.unique(D_neg, return_counts=True)
l_neg = l_neg / l_neg.sum()
w2id = {v: k for k, v in enumerate(words)}

_, priors = np.unique([k for _, k in D], return_counts=True)
N = priors.sum()
prior_pos = priors[1] / N
prior_neg = priors[0] / N

def likelihood(params, txt):
    params = np.log(params)
    _ = [params[w2id[x]] for x in txt.split()]
    tot = sum(_)
    return np.exp(tot)

post_pos = [likelihood(l_pos, x) * prior_pos for x, _ in D]
post_neg = [likelihood(l_neg, x) * prior_neg for x, _ in D]
evidence = np.vstack([post_pos, post_neg]).sum(axis=0)
post_pos /= evidence
post_neg /= evidence
hy = np.where(post_pos > post_neg, 1, 0)
y = np.array([y for _, y in D])

(hy == y).mean()

# Text Categorization - Naive Bayes

tm = TextModel(token_list=[-1], lang='english')
tok = tm.tokenize
D = [(tok(x['text']), x['klass']) for x in tweet_iterator(TWEETS)]
words = set()
[words.update(x) for x, y in D]
w2id = {v: k for k, v in enumerate(words)}
uniq_labels, priors = np.unique([k for _, k in D], return_counts=True)
priors = np.log(priors / priors.sum())
uniq_labels = {str(v): k for k, v in enumerate(uniq_labels)}
l_tokens = np.zeros((len(uniq_labels), len(w2id)))
for x, y in D:
    w = l_tokens[uniq_labels[y]]
    cnt = Counter(x)
    for i, v in cnt.items():
        w[w2id[i]] += v
l_tokens += 0.1
l_tokens = l_tokens / np.atleast_2d(l_tokens.sum(axis=1)).T
l_tokens = np.log(l_tokens)


def posteriori(txt):
    x = np.zeros(len(w2id))
    cnt = Counter(tm.tokenize(txt))
    for i, v in cnt.items():
        try:
            x[w2id[i]] += v
        except KeyError:
            continue
    _ = (x * l_tokens).sum(axis=1) + priors
    l = np.exp(_ - logsumexp(_))
    return l


hy = np.array([posteriori(x).argmax() for x, _ in D])
y = np.array([uniq_labels[y] for _, y in D])
(y == hy).mean()

# KFold and StratifiedKFold

def train(D):
    tok = tm.tokenize
    D =[(tok(x), y) for x, y in D]
    words = set()
    [words.update(x) for x, y in D]
    w2id = {v: k for k, v in enumerate(words)}
    uniq_labels, priors = np.unique([k for _, k in D], return_counts=True)
    priors = np.log(priors / priors.sum())
    uniq_labels = {str(v): k for k, v in enumerate(uniq_labels)}
    l_tokens = np.zeros((len(uniq_labels), len(w2id)))
    for x, y in D:
        w = l_tokens[uniq_labels[y]]
        cnt = Counter(x)
        for i, v in cnt.items():
            w[w2id[i]] += v
    l_tokens += 0.1
    l_tokens = l_tokens / np.atleast_2d(l_tokens.sum(axis=1)).T
    l_tokens = np.log(l_tokens)
    return w2id, uniq_labels, l_tokens, priors


D = [(x['text'], x['klass']) for x in tweet_iterator(TWEETS)]
tm = TextModel(token_list=[-1], lang='english')
folds = StratifiedKFold(shuffle=True, random_state=0)
hy = np.empty(len(D))
for tr, val in folds.split(D, y):
    training = [D[x] for x in tr]
    w2id, uniq_labels, l_tokens, priors = train(training)
    hy[val] = [posteriori(D[x][0]).argmax() for x in val]

y = np.array([uniq_labels[y] for _, y in D])
(y == hy).mean()
0.615


tm = TextModel(lang='english')
folds = StratifiedKFold(shuffle=True, random_state=0)
hy = np.empty(len(D))
for tr, val in folds.split(D, y):
    training = [D[x] for x in tr]
    w2id, uniq_labels, l_tokens, priors = train(training)
    assert np.all(np.isfinite([posteriori(D[x][0]) for x in val]))
    hy[val] = [posteriori(D[x][0]).argmax() for x in val]

y = np.array([uniq_labels[y] for _, y in D])
(y == hy).mean()
0.651

