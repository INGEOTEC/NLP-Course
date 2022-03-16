---
layout: default
title: Text Categorization
nav_order: 6
---

# Text Categorization
{: .fs-10 .no_toc }

## Contents
{: .no_toc .text-delta }

1. TOC
{:toc}

## Libraries used
{: .no_toc .text-delta }
```python
import numpy as np
from b4msa.textmodel import TextModel
from EvoMSA.tests.test_base import TWEETS
from microtc.utils import tweet_iterator
from scipy.stats import norm, multinomial
from matplotlib import pylab as plt
from collections import Counter
from sklearn.model_selection import StratifiedKFold
from scipy.special import logsumexp
from sklearn.metrics import recall_score, precision_score, f1_score
from EvoMSA.utils import bootstrap_confidence_interval
from sklearn.naive_bayes import MultinomialNB
```

## Installing external libraries
{: .no_toc .text-delta }

```bash
pip install b4msa
pip install evomsa
```

---

# Introduction

Text Categorization is an NLP task that deals with creating algorithms capable of identifying the category of a text from a set of predefined categories. For example, sentiment analysis belongs to this task, and the aim is to detect the polarity (e.g., positive, neutral, or negative) of a text. Furthermore, different NLP tasks that initially seem unrelated to this problem can be formulated as a classification one such as question answering and sentence entailment, to mention a few. 

Text Categorization can be tackled from different perspectives; the one followed here is to treat it as a supervised learning problem. As in any supervised learning problem, the starting point is a set of pairs, where the first element of the pair is the input and the second one corresponds to the output. Let $$\mathcal D = \{(\text{text}_i, y_i) \mid i=1,\ldots, N\}$$ where $$y \in \{c_1, \ldots c_K\}$$ and $$\text{text}_i$$ is a text. 

Supervised learning problems can be seen as finding a mapping function from inputs to outputs. The tool could be an [optimization](/NLP-Course/topics/02Vocabulary/#sec:optimization) algorithm capable of finding the function that minimizes a particular loss function, e.g., $$L$$. 

$$\min_{g \in \Omega} \sum_{(\mathbf x, y) \in \mathcal D} L(y, g(\mathbf x)),$$

where $$\Omega$$ is the search space of the feasible mapping functions.

Additionally, if one is also interested in measuring the uncertainty, the path relies on the probability. In this latter scenario, one approach is to assume the form of the conditional probability, i.e., $$\mathbb P(\mathcal Y=k \mid \mathcal X=x)=f_k(x)$$ where $$f_k$$ is the $$k$$-th value of $$f: \mathcal X \rightarrow [0, 1]^K$$ which encodes a probability mass function. For the case of a binary classification problem the function is $$f: \mathcal X \rightarrow [0, 1]$$. As can be seen, in this scenario, an adequate distribution is [Bernoulli](/NLP-Course/topics/03Collocations/#sec:bernoulli), where function $$f$$ takes the place of the parameter of the distribution, that is, $$\mathcal Y \sim \textsf{Bernoulli}(f(\mathcal X))$$; for more labels, the Categorical distribution can be used. On the other hand, the complement path is to rely on Bayes' theorem. 

# Bayes' theorem
{: #sec:bayes-theorem }

The bivariate distribution $$\mathbb P(\mathcal X, \mathcal Y)$$ can be expressed using the [conditional probability](/topics/04NGramLM/#sec:conditional-probability) as:

$$\begin{eqnarray}
\mathbb P(\mathcal X, \mathcal Y) &=& \mathbb P(\mathcal X \mid \mathcal Y) \mathbb P(\mathcal Y)\\
\mathbb P(\mathcal X, \mathcal Y) &=& \mathbb P(\mathcal Y \mid \mathcal X) \mathbb P(\mathcal X).
\end{eqnarray}$$

These elements can be combined to obtain Bayes' theorem following the next steps: 

$$\begin{eqnarray}
\mathbb P(\mathcal Y \mid \mathcal X) \mathbb P(\mathcal X) &=& \mathbb P(\mathcal X \mid \mathcal Y) \mathbb P(\mathcal Y)\\
\mathbb P(\mathcal Y \mid \mathcal X)  &=& \frac{\mathbb P(\mathcal X \mid \mathcal Y) \mathbb P(\mathcal Y)}{\mathbb P(\mathcal X)},
\end{eqnarray}$$

where $$\mathbb P(\mathcal Y \mid \mathcal X)$$ is the **posterior probability**, $$\mathbb P(\mathcal X \mid \mathcal Y)$$ corresponds to the **likelihood**, $$\mathbb P(\mathcal Y)$$ is the **prior**, and $$\mathbb P(\mathcal X)$$ is the **evidence**. The evidence can be expressed as $$\mathbb P(\mathcal X) = \sum_y \mathbb P(\mathcal X \mid \mathcal y) \mathbb P(\mathcal y)$$ which corresponds to the marginal $$\mathbb P(\mathcal X)$$ using the conditional probability; it can be observed that this term acts as a normalization constant.

The Bayes' theorem has two features that make it amenable to addressing classification problems. The first is that it is a generative model; besides tackling the classification problem, the model can be used to generate the data, i.e., the model learns the distribution of the dataset. 

The second characteristic is that the likelihood is a probability distribution given any class. Consequently, the problem is to estimate $$K$$ different distribution using the subset of the training set belonging to each different class. On the other hand, the prior is the estimated probability of each class, and the evidence can be estimated using the previous two values. 

## Example

In order to illustrate the process of computing the posterior, the following example uses two normals, each one corresponding to a different class; the red one is the negative class, and blue is used to depict the positive one.

```python
pos = norm(loc=3, scale=2.5)
neg = norm(loc=-0.5, scale=0.75)
```

![Two Normals](/NLP-Course/assets/images/two_normals.png)

The normal associated with the negative class is sampled 100 times; however, the sampled elements in the tail corresponding to a mass lower than 0.05 or higher than 0.95 are discarded. The distribution of the positive class is sampled 1000 times using the constraint that the points in the interval of the negative class are not considered.  

```python
_min = neg.ppf(0.05)
_max = neg.ppf(0.95)
D = [(x, 0) for x in neg.rvs(100) if x >= _min and x <= _max]
D += [(x, 1) for x in pos.rvs(1000) if x < _min or x > _max]
```

The following picture shows the distribution of the positive and negative classes; it can be observed that the two classes are separated by the constraints imposed. These points will be used to illustrate the procedure to estimate the posterior distribution given a dataset; the dataset is $$\mathcal D=\{(x_i, y_i \mid i=1, \ldots, N)\}$$ where $$x_i \in \mathbb R$$ and $$y_i \in \{0, 1\}$$.

![Two Normal Samples](/NLP-Course/assets/images/two_normal_samples.png)

The first step is to estimate the likelihood, i.e., $$\mathbb P(\mathcal X \mid \mathcal Y)$$ where $$\mathcal Y=1$$ and $$\mathcal Y=0$$. It is assumed that the likelihood is normally distributed; thus, it requires estimating the mean and the standard deviation, which can be done with the following code. 

```python
l_pos = norm(*norm.fit([x for x, k in D if k == 1]))
l_neg = norm(*norm.fit([x for x, k in D if k == 0]))
```

The second step is to compute the prior, i.e., $$\mathbb P(\mathcal Y)$$ which corresponds to estimating the parameters of a Categorical distribution; the following code relies on the use of `np.unique` to estimate them.

```python
_, priors = np.unique([k for _, k in D], return_counts=True)
N = priors.sum()
prior_pos = priors[1] / N
prior_neg = priors[0] / N
```

```
x = np.array([x for x, _ in D])
x.sort()
post_pos = l_pos.pdf(x) * prior_pos
post_neg = l_neg.pdf(x) * prior_neg

post = np.vstack([post_pos, post_neg])
evidence = post.sum(axis=0)
post_pos /= evidence
post_neg /= evidence
```

![Posterior of Two Classes](/NLP-Course/assets/images/two_classes_posterior.png)

![Posterior Errors](/NLP-Course/assets/images/two_classes_posterior_error.png)

# Categorical distribution

```python
m = {k: chr(122 - k) for k in range(4)}
pos = multinomial(1, [0.20, 0.20, 0.35, 0.25])
neg = multinomial(1, [0.35, 0.20, 0.25, 0.20])
length = norm(loc=10, scale=3)
D = []
id2w = lambda x: " ".join([m[_] for _ in x.argmax(axis=1)])
for l in length.rvs(size=1000):
    D.append((id2w(pos.rvs(round(l))), 1))
    D.append((id2w(neg.rvs(round(l))), 0))
```

|Text          |Label    |
|--------------|---------|
|x w x x z w y | 1       |
|y w z z z x w | 0       |
|z x x x z x z w x w | 1 |
|x w z w y z z z z w | 0 |


```python
D_pos = []
[D_pos.extend(data.split()) for data, k in D if k == 1]
D_neg = []
[D_neg.extend(data.split()) for data, k in D if k == 0]
```

```python
words, l_pos = np.unique(D_pos, return_counts=True)
w2id = {v: k for k, v in enumerate(words)}
l_pos = l_pos / l_pos.sum()
l_pos
array([0.25489421, 0.33854064, 0.20773186, 0.1988333 ])
```

```python
_, l_neg = np.unique(D_neg, return_counts=True)
l_neg = l_neg / l_neg.sum()
```

```python
_, priors = np.unique([k for _, k in D], return_counts=True)
N = priors.sum()
prior_pos = priors[1] / N
prior_neg = priors[0] / N
```

```python
def likelihood(params, txt):
    params = np.log(params)
    _ = [params[w2id[x]] for x in txt.split()]
    tot = sum(_)
    return np.exp(tot)
```

```python
post_pos = [likelihood(l_pos, x) * prior_pos for x, _ in D]
post_neg = [likelihood(l_neg, x) * prior_neg for x, _ in D]
evidence = np.vstack([post_pos, post_neg]).sum(axis=0)
post_pos /= evidence
post_neg /= evidence
hy = np.where(post_pos > post_neg, 1, 0)
```

# Accuracy

```python
y = np.array([y for _, y in D])
(hy == y).mean()
0.7215
```

# Confidence interval

```python
p = (hy == y).mean()
s = np.sqrt(p * (1 - p) / y.shape[0]) 
coef = norm.ppf(0.975)
ci = (p - coef * s, p + coef * s)
ci
(0.6845035761081213, 0.7244964238918787)
```

```python
ci = bootstrap_confidence_interval(y, hy, alpha=0.025,
                                  metric=lambda a, b: (a == b).mean())
ci                                  
(0.6842375, 0.7252625)
```

# Text Categorization - Naive Bayes

[tweets](https://raw.githubusercontent.com/INGEOTEC/EvoMSA/master/EvoMSA/tests/tweets.json)

```python
tm = TextModel(token_list=[-1], lang='english')
tok = tm.tokenize
```

```python
D = [(tok(x['text']), x['klass']) for x in tweet_iterator(TWEETS)]
```

```python
words = set()
[words.update(x) for x, y in D]
w2id = {v: k for k, v in enumerate(words)}
```

```python
uniq_labels, priors = np.unique([k for _, k in D], return_counts=True)
priors = np.log(priors / priors.sum())
uniq_labels = {str(v): k for k, v in enumerate(uniq_labels)}
```

```python
l_tokens = np.zeros((len(uniq_labels), len(w2id)))
for x, y in D:
    w = l_tokens[uniq_labels[y]]
    cnt = Counter(x)
    for i, v in cnt.items():
        w[w2id[i]] += v
l_tokens += 0.1
l_tokens = l_tokens / np.atleast_2d(l_tokens.sum(axis=1)).T
l_tokens = np.log(l_tokens)
```

```python
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
```

```python
hy = np.array([posteriori(x).argmax() for x, _ in D])
y = np.array([uniq_labels[y] for _, y in D])
(y == hy).mean()
0.977
```

# KFold and StratifiedKFold

```python
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
```

```python
p = (hy == y).mean()
s = np.sqrt(p * (1 - p) / y.shape[0]) 
coef = norm.ppf(0.975)
ci = (p - coef * s, p + coef * s)
ci
(0.5848410641389679, 0.6451589358610321)
```

# Precision, Recall, and F1-score

```python
p = precision_score(y, hy, average=None)
r = recall_score(y, hy, average=None)

2 * (p * r) / (p + r)
f1_score(y, hy, average=None)

metric = lambda a, b: recall_score(a, b, average='macro')
ci = bootstrap_confidence_interval(y, hy,
                                   metric=metric)
ci
(0.3967001014739609, 0.43613909993373073)
```

# Tokenizer

```python
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
```

```python
metric = lambda a, b: recall_score(a, b, average='macro')
ci = bootstrap_confidence_interval(y, hy,
                                   metric=metric)
ci
(0.42520668312638243, 0.4632802320946836)
```

