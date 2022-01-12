import numpy as np
from microtc.utils import tweet_iterator
from EvoMSA.tests.test_base import TWEETS
from text_models import Vocabulary
from text_models.utils import date_range
from collections import Counter
from matplotlib import pylab as plt
from scipy.optimize import minimize
from joblib import Parallel, delayed
from tqdm import tqdm
from wordcloud import WordCloud as WC
# %pylab inline

## Frequency of words

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

words = Counter()
for tw in tweet_iterator(TWEETS):
    text = tw['text']
    words.update([x.strip() for x in text.split()])

words['si']

# Zipf's Law

freq = [f for _, f  in words.most_common()]
rank = range(1, len(freq) + 1)
plt.plot(rank, freq, '.')
plt.grid()
plt.xlabel('Rank')
plt.ylabel('Frequency')
plt.tight_layout()
plt.savefig('zipf_law.png', dpi=300)


## Inverse rank

freq = [f for _, f  in words.most_common()]
rank = 1 / np.arange(1, len(freq) + 1)
plt.plot(rank, freq, '.')
plt.grid()
plt.xlabel('Inverse Rank')
plt.ylabel('Frequency')
plt.tight_layout()
plt.savefig('zipf_law2.png', dpi=300)

## OLS

X = np.atleast_2d(rank).T
c = np.linalg.lstsq(X, freq, rcond=None)[0]
c
hy = np.dot(X, c)
plt.plot(rank, freq, '.')
plt.plot(rank, hy)
plt.legend(['Measured', 'Predicted'])
plt.grid()
plt.xlabel('Inverse Rank')
plt.ylabel('Frequency')
plt.tight_layout()
plt.savefig('zipf_law3.png', dpi=300)

# Heaps' Law

words = Counter()
tokens_voc= list()
for tw in tweet_iterator(TWEETS):
    text = tw['text']
    words.update([x.strip() for x in text.split()])
    tokens_voc.append([sum(list(words.values())),
                       len(words)])

n = [x[0] for x in tokens_voc]
v = [x[1] for x in tokens_voc]
plt.plot(n, v, '.')
plt.grid()
plt.xlabel('Number of tokens')
plt.ylabel('Vocabulary Size')
plt.tight_layout()
plt.savefig('heaps_law.png', dpi=300)

## Optimization

n = np.array(n)
v = np.array(v)
def f(w, y, x):
    k, beta = w
    return ((y - k * x**beta)**2).sum()

res = minimize(f, np.array([1, 0.5]), (v, n))
k, beta = res.x
k, beta

plt.plot(n, v, '.')
plt.plot(n, k*n**beta)
plt.legend(['Measured', 'Predicted'])
plt.grid()
plt.xlabel('Number of tokens')
plt.ylabel('Vocabulary Size')
plt.tight_layout()
plt.savefig('heaps_law2.png', dpi=300)


# Activities
date = dict(year=2022, month=1, day=10)
voc = Vocabulary(date, lang='Es', country='MX')
words = {k: v for k, v in voc.voc.items() if not k.count('~')}

wc = WC().generate_from_frequencies(words)
plt.imshow(wc)
plt.axis('off')
plt.tight_layout()
plt.savefig('wordcloud_mx.png', dpi=300)

## Zipf's Law - $$f=\frac{c}{r}$$

countries = ['MX', 'CO', 'ES', 'AR',
             'PE', 'VE', 'CL', 'EC',
             'GT', 'CU', 'BO', 'DO', 
             'HN', 'PY', 'SV', 'NI', 
             'CR', 'PA', 'UY']
vocs = Parallel(n_jobs=-1)(delayed(Vocabulary)(date,
                                               lang='Es',
                                               country=country)
                               for country in tqdm(countries))
words = [{k: v for k, v in voc.voc.items() if not k.count('~')}
         for voc in vocs]

def zipf(data):
    freq = [f for _, f  in Counter(data).most_common()]
    rank = 1 / np.arange(1, len(freq) + 1)
    X = np.atleast_2d(rank).T
    return np.linalg.lstsq(X, freq, rcond=None)[0]

zipf_c = [zipf(w) for w in words]
tokens = [sum(list(w.values())) for w in words]


lst = [(a, b[0], c) for a, b, c in zip(countries, zipf_c, tokens)]
lst.sort(key=lambda x: x[1], reverse=True)

for a, b, c in lst:
    print("| {} | {:0.2f} | {:d} |".format(a, b,c))


X = np.array([(b[0], c) for b, c in zip(zipf_c, tokens)])
corr = np.corrcoef(X.T)


for c in corr:
    print("| {:0.4f} | {:0.4f} |".format(*c))


## Heaps' Law - $$\mid v \mid = kn^\beta$$

COUNTRIES = ['MX', 'CO', 'ES', 'AR',
             'PE', 'VE', 'CL', 'EC',
             'GT', 'CU', 'BO', 'DO', 
             'HN', 'PY', 'SV', 'NI', 
             'CR', 'PA', 'UY']

def get_words(date=dict(year=2022, month=1, day=10)):
   

    vocs = Parallel(n_jobs=-1)(delayed(Vocabulary)(date,
                                                   lang='Es',
                                                   country=country)
                                   for country in tqdm(COUNTRIES))
    words = [{k: v for k, v in voc.voc.items() if not k.count('~')}
             for voc in vocs]
    return words 


def voc_tokens(data):
    cnt = Counter(data[0])
    output = [[len(cnt), sum(list(cnt.values()))]]
    for x in data[1:]:
        cnt.update(x)
        _ = [len(cnt), sum(list(cnt.values()))]
        output.append(_)
    output = np.array(output)
    return output[:, 0], output[:, 1]

init = dict(year=2021, month=11, day=1)
end = dict(year=2021, month=11, day=30)
dates = date_range(init, end)
words = [get_words(d) for d in dates]
ww = [[w[index] for w in words] for index in range(len(COUNTRIES))]

n_mx, v_mx = voc_tokens(ww[0])
