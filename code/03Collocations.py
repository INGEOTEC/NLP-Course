from text_models import Vocabulary
from collections import Counter
import numpy as np
from scipy.stats import norm, chi2
from wordcloud import WordCloud as WC
from matplotlib import pylab as plt
from collections import defaultdict
# %pylab inline

# Collocations

date = dict(year=2022, month=1, day=10)
voc = Vocabulary(date, lang='En', country="US")
bigrams = Counter({k: v for k, v in voc.voc.items() if k.count("~")})
co_occurrence = np.zeros((5, 5))
index = {'the': 0, 'to': 1, 'of': 2, 'in': 3, 'and': 4}

for bigram, cnt in bigrams.most_common():
    a, b = bigram.split('~')
    if a in index and b in index:
        co_occurrence[index[a], index[b]] = cnt
        co_occurrence[index[b], index[a]] = cnt

keys = list(index.items())
keys.sort(key=lambda x: x[1])

print(' | ' + ' | '.join([k for k, _ in keys]) + ' | ')
for c, (k, _) in zip(co_occurrence, keys):
    _ = " | ".join(map(lambda x: '{: 7d}'.format(int(x)), c))
    print('{} | {} |'.format(k, _))


for bigram, cnt in bigrams.items():
    a, b = bigram.split('~')
    for x in [a, b]:
        if x not in index:
            index[x] = len(index)
len(index)

# Bernoulli Distribution
x = np.random.binomial(1, 0.3, size=1000)
hp = x.mean()

# Categorical distribution
X = np.random.multinomial(1, [1/6] * 6, size=100)
x = X.argmax(axis=1)

var, counts = np.unique(x, return_counts=True)
N = counts.sum()
p = counts / N

# Bivariate distribution
co_occurrence = np.zeros((len(index), len(index)))
for bigram, cnt in bigrams.most_common():
    a, b = bigram.split('~')
    if a in index and b in index:
        co_occurrence[index[a], index[b]] = cnt
        co_occurrence[index[b], index[a]] = cnt
co_occurrence = co_occurrence / co_occurrence.sum()

keys = list(index.items())
keys.sort(key=lambda x: x[1])

print(' | ' + ' | '.join([k for k, _ in keys[:5]]) + ' | ')
for c, (k, _) in zip(co_occurrence, keys[:5]):
    _ = " | ".join(map(lambda x: '{: 0.5f}'.format(x), c[:5]))
    print('{} | {} |'.format(k, _))


# Independent Random Variables

d = 6
R = np.random.multinomial(1, [1/d] * d, size=10000).argmax(axis=1)
C = np.random.multinomial(1, [1/d] * d, size=10000).argmax(axis=1)
Z = [[r, c] for r, c in zip(R, C)]
Z = [[r, c] for r, c in zip(R, C) if r != c]
Z = [[2 if c == 1 and np.random.rand() < 0.1 else r, c] for r, c in zip(R, C)]

W = np.zeros((d, d))
for r, c in Z:
    W[r, c] += 1
W = W / W.sum()

for w in (W):
    _ = " & ".join(map(lambda x: "{:0.4f}".format(x), w))
    print(r"{} \\".format(_))

R_m = W.sum(axis=1)
C_m = W.sum(axis=0)
ind = np.dot(np.atleast_2d(R_m).T, np.atleast_2d(C_m))

for w in (W-ind):
    _ = " & ".join(map(lambda x: "{:0.4f}".format(x), w))
    print(r"{} \\".format(_))

# Example of the [bigrams](#tab:bivariate-distribution)

_ = [(bigram, [index[x] for x in bigram.split("~")]) for bigram in bigrams]
_ = {key: co_occurrence[i, j] for key, (i, j) in _}
wc = WC().generate_from_frequencies(_)
plt.imshow(wc)
plt.axis('off')
plt.tight_layout()
plt.savefig('wordcloud_us.png', dpi=300)

M = co_occurrence.sum(axis=1)

def get_diff(key):
    a, b = [index[x] for x in key.split('~')]
    if a == b:
        return - M[a] * M[b]    
    return co_occurrence[a, b] - M[a] * M[b]


print(' | ' + ' | '.join([k for k, _ in keys[:5]]) + ' | ')
for k, _ in keys[:5]:
    values = [get_diff("{}~{}".format(k, j)) for j, _ in keys[:5]]
    _ = " | ".join(map(lambda x: '{: 0.5f}'.format(x), values))
    print('{} | {} |'.format(k, _))

freq = {x: get_diff(x) for x in bigrams.keys()}
freq = {k: v for k, v in freq.items() if v > 0}

wc = WC().generate_from_frequencies(freq)
plt.imshow(wc)
plt.axis('off')
plt.tight_layout()
plt.savefig('wordcloud_us2.png', dpi=300)

## The Wald Test

N = len(Z)
se = np.sqrt(W * (1 - W) / N)

wald = (W - ind) / se 
for w in (wald):
    _ = " & ".join(map(lambda x: "{:0.4f}".format(x), w))
    print(r"{} \\".format(_))


N = sum(list(bigrams.values()))
se = lambda x: np.sqrt(x * (1 - x) / N)
_ = [(bigram, [index[x] for x in bigram.split("~")]) for bigram in bigrams]
co = co_occurrence
wald = {k: (co[i, j] - M[i] * M[j]) / se(co[i, j])
        for k, (i, j) in _}

alpha = 0.01
c = norm.ppf(1 - alpha / 2)
wald = {k: v for k, v in wald.items() if v > c}

wc = WC().generate_from_frequencies(wald)
plt.imshow(wc)
plt.axis('off')
plt.tight_layout()
plt.savefig('wordcloud_us3.png', dpi=300)


## Likelihood ratios
count = dict()
for k, v in bigrams.items():
    for x in k.split('~'):
        try:
            count[x] += v
        except KeyError:
            count[x] = v 

N = sum(list(count.values()))

def L(k, n, x):
    f1 = k * np.log(x)
    f2 = (n - k) * np.log(1 - x)
    return f1 + f2


def ratio(k):
    a, b = k.split('~')
    c12 = bigrams[k]
    c1 = count[a]
    c2 = count[b]
    p = c2 / N
    p1 = c12 / c1
    p2 = (c2 - c12) / (N - c1)
    f1 = L(c12, c1, p) +  L(c2 - c12, N - c1, p)
    f2 = L(c12, c1, p1) + L(c2 - c12, N - c1, p2)
    return -2 * (f1 - f2)

r = {k: ratio(k) for k, v in bigrams.items()}
c = chi2.ppf((1 - alpha), 1)
r = {k: v for k, v in r.items() if np.isfinite(v) and v > c}

wc = WC().generate_from_frequencies(r)
plt.imshow(wc)
plt.axis('off')
plt.tight_layout()
plt.savefig('wordcloud_us4.png', dpi=300)

# Activities

scatter = [[bigrams[k], v] for k, v in wald.items()]
plt.semilogx([x for x, _ in scatter],
             [y for _, y in scatter], '.')
plt.grid()
plt.xlabel('Frequency')
plt.ylabel('Wald Statistic')
plt.tight_layout()
plt.savefig('scatter_plot_hypw.png', dpi=300)


plt.rcParams['text.usetex'] = True
scatter = [[bigrams[k], v] for k, v in r.items()]
plt.loglog([x for x, _ in scatter],
           [y for _, y in scatter], '.')
plt.xlabel('Frequency')
plt.ylabel(r'$-2 \log \lambda$')
plt.grid()
plt.tight_layout()
plt.savefig('scatter_plot_hypl.png', dpi=300)


date = dict(year=2022, month=1, day=17)
voc = Vocabulary(date, lang='En', country="US")
bigrams = Counter({k: v for k, v in voc.voc.items() if k.count("~")})

d = [(k, v) for k, v in r.items()]
d.sort(key=lambda x: x[1], reverse=True)

_ = {k: v for k, v in d[200:400]}
wc = WC().generate_from_frequencies(_)
plt.imshow(wc)
plt.axis('off')
plt.tight_layout()
plt.savefig('wordcloud_us6.png', dpi=300)
