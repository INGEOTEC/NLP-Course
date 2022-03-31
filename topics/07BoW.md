---
layout: default
title: Bag of Words Model
nav_order: 7
---

# Bag of Words Model
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
from microtc.utils import tweet_iterator, load_model, save_model
from scipy.stats import norm, multinomial, multivariate_normal
from matplotlib import pylab as plt
from collections import Counter
from sklearn.model_selection import StratifiedKFold
from scipy.special import logsumexp
from sklearn.metrics import recall_score, precision_score, f1_score
from EvoMSA.utils import bootstrap_confidence_interval
from sklearn.naive_bayes import MultinomialNB
from os.path import join
```

## Installing external libraries
{: .no_toc .text-delta }

```bash
pip install b4msa
pip install evomsa
```

---

# Introduction

The problem of text categorization can be tackled directly by modeling the conditional probability, i.e., $$\mathbb P(\mathcal Y=k \mid \mathcal X=x)=f_k(x)$$ where $$f_k$$ is the $$k$$-th value of $$f: \mathcal X \rightarrow [0, 1]^K$$ which encodes a probability mass function. For the case of $$K$$ classes an adequate distribution is [Categorical](/NLP-Course/topics/03Collocations/#sec:categorical), where function $$f$$ takes the place of the parameter of the distribution, that is, $$\mathcal Y \sim \textsf{Categorical}(f(\mathcal X))$$.

# Maximum Likelihood Estimator

$$l_{f_\mathcal X}(f) = \log \prod_{(x, y) \in \mathcal D} \prod_{k=1}^K f_k(x)^{\delta(k=y)}$$ 

$$\frac{\partial}{\partial w_j} l_{f_\mathcal X}(f) = \frac{\partial}{\partial w_j} \log \prod_{(x, y) \in \mathcal D} \prod_{k=1}^K f_k(x)^{\delta(k=y)}$$

$$ = \sum_{(x, y) \in \mathcal D} \sum_{k=1}^K  \delta(k=y) \frac{\partial}{\partial w_j} \log f_k(x)$$

$$ = \sum_{(x, y) \in \mathcal D} \sum_{k=1}^K  \frac{\delta(k=y)}{f_k(x)} \frac{\partial}{\partial w_j} f_k(x)$$

---

$$\sum_k^K f_k(x) = 1$$

$$f_k(x) = \frac{h_k(w_k(x))}{\sum_{\ell=1}^K h_\ell(w_\ell(x))}$$


$$\frac{\partial}{\partial w_j} f_k(x) = \frac{\partial}{\partial w_j} \frac{h_k(w_k(x))}{\sum_{\ell=1}^K h_\ell(w_\ell(x))} $$

$$ = \frac{\sum_{\ell=1}^K h_\ell(w_\ell(x)) \frac{\partial}{\partial w_j} h_k(w_k(x)) - h_k(w_k(x)) \frac{\partial}{\partial w_j} \sum_{\ell=1}^K h_\ell(w_\ell(x))}{(\sum_{\ell=1}^K h_\ell(w_\ell(x)))^2} $$

$$ = \frac{\sum_{\ell=1}^K h_\ell(w_\ell(x)) \frac{\partial}{\partial w_j} h_k(w_k(x)) - h_k(w_k(x)) h_j(w_j(x)) \frac{\partial}{\partial w_j} w_j(x)}{(\sum_{\ell=1}^K h_\ell(w_\ell(x)))^2} $$

