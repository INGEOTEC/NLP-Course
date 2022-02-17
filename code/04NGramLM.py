import numpy as np
from matplotlib import pylab as plt
from microtc.utils import tweet_iterator
from os.path import join
from collections import Counter, defaultdict
from wordcloud import WordCloud as WC
from EvoMSA.utils import bootstrap_confidence_interval
# %pylab inline

plt.rcParams['text.usetex'] = True
d = 8
l = range(1, 5)
plt.plot(l, [d**i for i in l])
plt.xlabel(r'$\ell$')
plt.ylabel(r'$d^\ell$')
plt.grid()
plt.tight_layout()


# Bigrams 
d = 4
R = np.random.multinomial(1, [1/d] * d, size=10000).argmax(axis=1)
C = np.random.multinomial(1, [1/d] * d, size=10000).argmax(axis=1)
rand = np.random.rand
Z = [[r, 2 if r == 1 and rand() < 0.1 else c]
      for r, c in zip(R, C)
     if r != c or (r == c and rand() < 0.2)]

W = np.zeros((d, d))
for r, c in Z:
    W[r, c] += 1
W = W / W.sum()

for w in W:
    _ = " & ".join(map(lambda x: "{:0.4f}".format(x), w))
    print(r"{} \\".format(_))

M_r = W.sum(axis=1)
_ = map(lambda x: "{:4f}".format(x), W.sum(axis=0))
print(", ".join(_))
p_l = (W / np.atleast_2d(M_r).T)

for w in p_l:
    _ = " & ".join(map(lambda x: "{:0.4f}".format(x), w))
    print(r"{} \\".format(_))

## Generating sequences

cat = lambda x: np.random.multinomial(1, x, 1).argmax()
id2word = {0: 'a', 1: 'b', 2: 'c', 3: 'd'}
w1 = cat(M_r)

text = [cat(M_r)]
l = 25
while len(text) < l:
    next = cat(p_l[text[-1]])
    text.append(next)
text = " ".join(map(lambda x: id2word[x], text))
text


w2id = {v: k for k, v in id2word.items()}
lst = [w2id[x] for x in text.split()]
Z = [[a, b] for a, b in zip(lst, lst[1:])]

d = len(w2id)
W = np.zeros((d, d))
for r, c in Z:
    W[r, c] += 1
W = W / W.sum()

for w in W:
    _ = " & ".join(map(lambda x: "{:0.4f}".format(x), w))
    print(r"{} \\".format(_))


text = 'a d b c'
lst = [w2id[x] for x in text.split()]
p = M_r[lst[0]]
for a, b in zip(lst, lst[1:]):
    p *= p_l[a, b]


# Bigram LM from Tweets    

fname = join('dataset', 'tweets-2022-01-17.json.gz')
bigrams = Counter()
for text in tweet_iterator(fname):
    text = text['text']
    words = text.split()
    words.insert(0, '<s>')
    words.append('</s>')
    _ = [(a, b) for a, b in zip(words, words[1:])]
    bigrams.update(_)

words = set([a for a, b in bigrams])
words2 = set([b for a, b in bigrams])

prev = dict()
for (a, b), v in bigrams.items():
    try:
        prev[a] += v
    except KeyError:
        prev[a] = v

wc = WC().generate_from_frequencies(prev)
plt.imshow(wc)
plt.axis('off')
plt.tight_layout()

P = defaultdict(Counter)
for (a, b), v in bigrams.items():
    next = P[a]
    next[b] = v / prev[a]

wc = WC().generate_from_frequencies(P['<s>'])
plt.imshow(wc)
plt.axis('off')
plt.tight_layout()
plt.savefig('wordcloud_prob_start.png', dpi=300)

sentence = ['<s>']
while sentence[-1] != '</s>':
    var = P[sentence[-1]]
    pos = var.most_common(20)
    index = np.random.randint(len(pos))
    sentence.append(pos[index][0])
len(sentence)


def joint_prob(sentence):
    words = sentence.split()
    words.insert(0, '<s>')
    words.append('</s>')
    tot = 1
    for a, b in zip(words, words[1:]):
        tot *= P[a][b]
    return tot

joint_prob('I like to play football')
joint_prob('I like to play soccer')


# Performance


def PP(sentences,
       prob=lambda a, b: P[a][b]):
    if isinstance(sentences, str):
        sentences = [sentences]
    tot, N = 0, 0
    for sentence in sentences:
        words = sentence.split()
        words.insert(0, '<s>')
        words.append('</s>')
        tot = 0
        for a, b in zip(words, words[1:]):
            tot += np.log(1 / prob(a, b))
        N += (len(words) - 1)
    _ = tot / (len(words) - 1)
    return np.exp(_)


text = 'I like to play football'
PP(text)

fname2 = join('dataset', 'tweets-2022-01-17.json.gz')
PP([x['text'] for x in tweet_iterator(fname2)])

PP('I like to play soccer')

## Laplace Smoothing

V = set()
_ = [[V.add(x) for x in key] for key in bigrams.keys()]
V = len(V) + 1

prev_l = dict()
for (a, b), v in bigrams.items():
    try:
        prev_l[a] += v
    except KeyError:
        prev_l[a] = v

P_l = defaultdict(Counter)
for (a, b), v in bigrams.items():
    next = P_l[a]
    next[b] = (v + 1) / (prev_l[a] + V)

for (w, a), (_, b) in zip(P['<s>'].most_common(4),
                          P_l['<s>'].most_common(4)):
    print("|{}|{:4f}|{:4f}|".format(w, a, b))


K = 1 
def laplace(a, b):
    if a in P_l:
        next = P_l[a]
        if b in next:
            return next[b]
    if a in prev_l:
        return K / (prev_l[a] + K * V)
    return 1 / V


fname2 = join('dataset', 'tweets-2022-01-10.json.gz')
PP([x['text'] for x in tweet_iterator(fname2)],
    prob=laplace)

# Activities

## Add-$$k$$ Smoothing

def cond_prob(ngrams, prev):
    output = defaultdict(Counter)
    for (*a, b), v in ngrams.items():
        key = tuple(a)
        next = output[key]
        next[b] = (v + K) / (prev[key] + K * V)
    return output


def sum_last(data):
    output = Counter()
    for (*prev, last), v in data.items():
        key = tuple(prev)
        output.update({key: v})
    return output


prev_l = sum_last(bigrams)
K = 0.1
P_l = cond_prob(bigrams, prev_l)
PP('I like to play soccer', 
   prob=lambda a, b: laplace((a, ), b))

fname2 = join('dataset', 'tweets-2022-01-10.json.gz')

output = []
for k in np.linspace(0.01, 1, 10):
    K = k
    P_l = cond_prob(bigrams, prev_l)
    _ = PP([x['text'] for x in tweet_iterator(fname2)], 
            prob=lambda a, b: laplace((a, ), b))
    output.append(_)


# one=output when the file is 'tweets-2022-01-17.json.gz'
x = np.linspace(0.01, 1, 10)
plt.plot(x, one)
plt.plot(x, output)
plt.grid()
plt.xlabel(r'$k$')
plt.ylabel(r'$PP$')
plt.legend(['Training', 'Test'])
plt.tight_layout()
plt.savefig('laplace_smoothing.png', dpi=300)

## Max Smoothing

def sum_last_max(data):
    tokens = Counter()
    output = Counter()
    for (*prev, last), v in data.items():
        key = tuple(prev)
        output.update({key: v})
        tokens.update({key: 1})
    for key, v in tokens.items():
        output.update({key: K * (V - v)})
    return output


def cond_prob_max(ngrams, prev):
    output = defaultdict(Counter)
    for (*a, b), v in ngrams.items():
        key = tuple(a)
        next = output[key]
        next[b] = v / prev[key]
    return output 


def prob_max(a, b):
    if a in P_l:
        next = P_l[a]
        if b in next:
            return next[b]
    if a in prev_l:
        return K / prev_l[a]
    return 1 / V

K = 0.1
prev_l = sum_last_max(bigrams)
P_l = cond_prob_max(bigrams, prev_l)
PP('I like to play soccer', 
   prob=lambda a, b: prob_max((a, ), b))

fname2 = join('dataset', 'tweets-2022-01-10.json.gz')

output = []
for k in np.linspace(0.01, 1, 10):
    K = k
    prev_l = sum_last_max(bigrams)
    P_l = cond_prob_max(bigrams, prev_l)
    _ = PP([x['text'] for x in tweet_iterator(fname2)], 
            prob=lambda a, b: prob_max((a, ), b))
    output.append(_)


# one=output when the file is 'tweets-2022-01-17.json.gz'
x = np.linspace(0.01, 1, 10)
plt.plot(x, one)
plt.plot(x, output)
plt.grid()
plt.xlabel(r'$k$')
plt.ylabel(r'$PP$')
plt.legend(['Training', 'Test'])
plt.tight_layout()
plt.savefig('max_smoothing.png', dpi=300)

## N-Gram

def compute_ngrams(fname, n=3):
    ngrams = Counter()
    for text in tweet_iterator(fname):
        text = text['text']
        words = text.split()
        [words.insert(0, '<s>') for _ in range(n - 1)]
        words.append('</s>')
        _ = [a for a in zip(*(words[i:] for i in range(n)))]
        ngrams.update(_)
    return ngrams


fname = join('dataset', 'tweets-2022-01-17.json.gz')
ngrams = compute_ngrams(fname, n=2)
V = set()
_ = [[V.add(x) for x in key] for key in ngrams.keys()]
prev_l = sum_last(ngrams)
P_l = cond_prob(ngrams, prev_l, k=1)


def PP(sentences,
       prob=lambda a, b: P_l[a][b], n=3):
    if isinstance(sentences, str):
        sentences = [sentences]
    tot, N = 0, 0
    for sentence in sentences:
        words = sentence.split()
        [words.insert(0, '<s>') for _ in range(n-1)]
        words.append('</s>')
        tot = 0
        for *a, b in zip(*(words[i:] for i in range(n))):
            tot += np.log(1 / prob(tuple(a), b))
        N += (len(words) - (n - 1))
    _ = tot / (len(words) - (n - 1))
    return np.exp(_)

### 

