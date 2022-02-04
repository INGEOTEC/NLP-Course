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


def joint_prob(sentence):
    words = sentence.split()
    words.insert(0, '<s>')
    words.append('</s>')
    tot = 1
    for a, b in zip(words, words[1:]):
        tot *= P[a][b]
    return tot

joint_prob('I like to play football')
joint_prob('I like to dance')


sentence = ['<s>']
while sentence[-1] != '</s>':
    var = P[sentence[-1]]
    pos = var.most_common(20)
    index = np.random.randint(len(pos))
    sentence.append(pos[index][0])
len(sentence)

sentence = ['<s>']
for _ in range(20):
    var = P[sentence[-1]]
    pos = var.most_common(1)
    sentence.append(pos[0][0])

# Performance


def PP(sentence):
    words = sentence.split()
    words.insert(0, '<s>')
    words.append('</s>')
    tot = 0
    for a, b in zip(words, words[1:]):
        tot += np.log(1 / P[a][b])
    _ = tot / (len(words) - 1)
    return np.exp(_)


def PP(sentences):
    if isinstance(sentences, str):
        sentences = [sentences]
    tot, N = 0, 0
    for sentence in sentences:
        words = sentence.split()
        words.insert(0, '<s>')
        words.append('</s>')
        tot = 0
        for a, b in zip(words, words[1:]):
            tot += np.log(1 / P[a][b])
        N += (len(words) - 1)
    _ = tot / (len(words) - 1)
    return np.exp(_)


text = 'I like to play football'
PP(text)

fname2 = join('dataset', 'tweets-2022-01-17.json.gz')
PP([x['text'] for x in tweet_iterator(fname2)])