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

The parameters of the function $$f$$ can be identified using a maximum likelihood estimator; this procedure is equivalent to the one used for parameter $$\mathbf p$$ of the [Categorical](/NLP-Course/topics/03Collocations/#sec:categorical) distribution.

# Maximum Likelihood Estimator

The log-likelihood estimator is defined as follows, where $$\mathcal D$$ is the dataset used to estimate the parameters, $$f_\mathcal Y$$ is the probability mass function that corresponds to the Categorical distribution, and $$f(x)$$ takes the place of the distribution parameter; as can be observed, it is a function of the input. 

$$\begin{eqnarray}
l_{f_\mathcal Y}(f) &=& \log \prod_{(x, y) \in \mathcal D} f_\mathcal Y(y \mid f(x)) \\
&=& \log \prod_{(x, y) \in \mathcal D} \prod_{k=1}^K f_k(x)^{\mathbb 1(k=y)}\\
\end{eqnarray}$$

Assuming that function $$f$$ has a parameter $$w_j$$, the procedure to estimate the parameter is to compute the partial derivate of the log-likelihood with respect to $$w_j$$ and solve it when it is equal to zero. 

$$\begin{eqnarray}
\frac{\partial}{\partial w_j} l_{f_\mathcal Y}(f) &=& \frac{\partial}{\partial w_j} \log \prod_{(x, y) \in \mathcal D} \prod_{k=1}^K f_k(x)^{\mathbb 1(k=y)}\\
&=& \frac{\partial}{\partial w_j} \sum_{(x, y) \in \mathcal D} \sum_{k=1}^K \mathbb 1(k=y) \log f_k(x) = 0
\end{eqnarray}$$

# Minimizing Cross-entropy

Before solving the log-likelihood, it is essential to relate this concept with cross-entropy. First the expectation of $$h(\mathcal X)$$ is computed as $$\sum_x h(x) f(x)$$ where $$f$$ is the mass function, this can be expressed as $$\mathbb E_f[h(\mathcal X)]$$. On the other hand, the information content of an event is a decreasing function that has its zero when the event has the highest probability, meaning that there is no information carried on an event that occurs always. The information content can be modeled with the function $$I_f(e) = \log(\frac{1}{f(e)})=-\log(f(e)).$$  The **entropy** measures the expected value of the information content that is $$\mathbb E_f[I_f(\mathcal X)]=-\sum_x f(x) \log(f(x)).$$ Finally, the **cross-entropy** between distribution $$p$$ and $$q$$ is defined as $$H(p, q) = \mathbb E_p[I_q(\mathcal X)] = -\sum_x p(x) \log(q(x)).$$ 


It can be observed that the negative of the log-likelihood is accumulating the cross-entropy for all the elements in the dataset $$\mathcal D$$; the variables $$y$$ and $$x$$ are constants in the inner summation, and variable $$k$$ goes from all the classes. Therefore, minimizing the log-likelihood is minimizing the cross-entropy, which acts as the loss function. 

$$\begin{eqnarray}
-\frac{\partial}{\partial w_j}  l_{f_\mathcal Y}(f) &=& \frac{\partial}{\partial w_j} \sum_{(x, y) \in \mathcal D} \overbrace{- \sum_{k=1}^K \mathbb 1(k=y) \log f_k(x)}^{cross-entropy} \\
&=& -\sum_{(x, y) \in \mathcal D} \sum_{k=1}^K  \mathbb 1(k=y) \frac{\partial}{\partial w_j} \log f_k(x)\\
&=& - \sum_{(x, y) \in \mathcal D} \sum_{k=1}^K  \frac{\mathbb 1(k=y)}{f_k(x)} \frac{\partial}{\partial w_j} f_k(x)
\end{eqnarray}$$



$$\sum_k^K f_k(x) = 1$$

$$\begin{eqnarray}
f_k(x) &=& \frac{h_k(w_k(x))}{\sum_{\ell=1}^K h_\ell(w_\ell(x))}\\
\frac{\partial}{\partial w_j} f_k(x) &=& \frac{\partial}{\partial w_j} \frac{h_k(w_k(x))}{\sum_{\ell=1}^K h_\ell(w_\ell(x))}\\
&=& \frac{\sum_{\ell=1}^K h_\ell(w_\ell(x)) \frac{\partial}{\partial w_j} h_k(w_k(x)) - h_k(w_k(x)) \frac{\partial}{\partial w_j} \sum_{\ell=1}^K h_\ell(w_\ell(x))}{(\sum_{\ell=1}^K h_\ell(w_\ell(x)))^2}\\
&=& \frac{\sum_{\ell=1}^K h_\ell(w_\ell(x)) \frac{\partial}{\partial w_j} h_k(w_k(x)) - h_k(w_k(x)) \frac{\partial}{\partial w_j} h_j(w_j(x))}{(\sum_{\ell=1}^K h_\ell(w_\ell(x)))^2} 
\end{eqnarray}$$

$$\begin{eqnarray}
-\frac{\partial}{\partial w_j}  l_{f_\mathcal Y}(f) &=& - \sum_{(x, y) \in \mathcal D} \sum_{k=1}^K  \frac{\mathbb 1(k=y)}{f_k(x)} \frac{\partial}{\partial w_j} f_k(x)\\
&=& - \sum_{(x, y) \in \mathcal D} \sum_{k=1}^K \frac{\mathbb 1(k=y)}{h_k(w_k(x))}   \frac{\sum_{\ell=1}^K h_\ell(w_\ell(x)) \frac{\partial}{\partial w_j} h_k(w_k(x)) - h_k(w_k(x)) \frac{\partial}{\partial w_j} h_j(w_j(x))}{\sum_{\ell=1}^K h_\ell(w_\ell(x))}
\end{eqnarray}$$

# Softmax Function
$$\begin{eqnarray}
h_k(w_k(x)) &=& \exp(w_k(x))\\
f_k(w_k(x)) &=& \frac{\exp(w_k(x))}{\sum_\ell \exp(w_\ell(x))}
\end{eqnarray}$$


$$\begin{eqnarray}
-\frac{\partial}{\partial w_j}  l_{f_\mathcal Y}(f) &=&  - \sum_{(x, y) \in \mathcal D} \sum_{k=1}^K  \frac{\mathbb 1(k=y)}{\exp(w_k(x))} \frac{\sum_{\ell=1}^K \exp(w_\ell(x)) \frac{\partial}{\partial w_j} \exp(w_k(x)) - \exp(w_k(x)) \exp(w_j(x)) \frac{\partial}{\partial w_j} w_j(x)}{\sum_{\ell=1}^K \exp(w_\ell(x))}\\
&=& - \sum_{(x, y) \in \mathcal D} \sum_{k=1}^K  \frac{\mathbb 1(k=y)}{\exp(w_k(x))} \frac{\sum_{\ell=1}^K \exp(w_\ell(x)) \exp(w_k(x)) \frac{\partial}{\partial w_j} w_k(x) - \exp(w_k(x)) \exp(w_j(x)) \frac{\partial}{\partial w_j} w_j(x)}{\sum_{\ell=1}^K \exp(w_\ell(x))}\\
&=& - \sum_{(x, y) \in \mathcal D} \sum_{k=1}^K  \mathbb 1(k=y) \frac{\sum_{\ell=1}^K \exp(w_\ell(x))  \frac{\partial}{\partial w_j} w_k(x) - \exp(w_j(x)) \frac{\partial}{\partial w_j} w_j(x)}{\sum_{\ell=1}^K \exp(w_\ell(x))}\\
&=& - \sum_{(x, y) \in \mathcal D} \sum_{k=1}^K  \mathbb 1(k=y) \left[ \frac{\partial}{\partial w_j} w_k(x) - f_j(x) \frac{\partial}{\partial w_j} w_j(x)\right]\\
&=& - \sum_{(x, y) \in \mathcal D}  \mathbb 1(j=y) \left[ \frac{\partial}{\partial w_j} w_j(x) - f_j(x) \frac{\partial}{\partial w_j} w_j(x)\right] + \sum_{k \neq j}^K  \mathbb 1(k=y) \left[ \frac{\partial}{\partial w_j} w_k(x) - f_j(x) \frac{\partial}{\partial w_j} w_j(x)\right]\\
&=& - \sum_{(x, y) \in \mathcal D}  \mathbb 1(j=y) \left[ 1 - f_j(x) \right] \frac{\partial}{\partial w_j} w_j(x) + \sum_{k \neq j}^K  -\mathbb 1(k=y) f_j(x) \frac{\partial}{\partial w_j} w_j(x)\\
&=& - \sum_{(x, y) \in \mathcal D}  \left( \mathbb 1(j=y) - f_j(x) \right) \frac{\partial}{\partial w_j} w_j(x)\\
&=& \sum_{(x, y) \in \mathcal D}  \left( f_j(x) - \mathbb 1(j=y) \right) \frac{\partial}{\partial w_j} w_j(x)
\end{eqnarray}$$

# Logistic Regression

$$\begin{eqnarray}
f(x) &=& \frac{1}{1 + \exp(-x)}\\
f_1(x) &=& f(w(x)) \\
f_2(x) &=& 1 - f_1(x) 
\end{eqnarray}$$

$$\begin{eqnarray}
-\frac{\partial}{\partial w_j}  l_{f_\mathcal Y}(f) &=& - \sum_{(x, y) \in \mathcal D} \sum_{k=1}^K  \frac{\mathbb 1(k=y)}{f_k(x)} \frac{\partial}{\partial w_j} f_k(x)\\
&=& - \sum_{(x, y) \in \mathcal D} \frac{\mathbb 1(1=y)}{f_1(x)} \frac{\partial}{\partial w_j} f_1(x) + \frac{\mathbb 1(2=y)}{1 - f_1(x)} \frac{\partial}{\partial w_j} \left(  1 - f_1(x) \right)\\
&=& - \sum_{(x, y) \in \mathcal D} \frac{\mathbb 1(1=y)}{f(w(x))} \frac{\partial}{\partial w_j} f(w(x)) - \frac{\mathbb 1(2=y)}{1 - f(w(x))} \frac{\partial}{\partial w_j} f(w(x))\\
&=& - \sum_{(x, y) \in \mathcal D} \left[ \frac{\mathbb 1(1=y)}{f(w(x))} - \frac{\mathbb 1(2=y)}{1 - f(w(x))} \right] \frac{\partial}{\partial w_j} f(w(x))\\
&=& - \sum_{(x, y) \in \mathcal D} \left[ \frac{\mathbb 1(1=y)}{f(w(x))} - \frac{\mathbb 1(2=y)}{1 - f(w(x))} \right] (1 - f(w(x))) f(w(x)) \frac{\partial}{\partial w_j} w(x)\\
&=& - \sum_{(x, y) \in \mathcal D} \left[ (1 - f(w(x)))\mathbb 1(1=y) - f(w(x))\mathbb 1(2=y) \right] \frac{\partial}{\partial w_j} w(x)\\
&=& - \sum_{(x, y) \in \mathcal D} (\mathbb 1(1=y) - f(w(x))) \frac{\partial}{\partial w_j} w(x)\\
&=& \sum_{(x, y) \in \mathcal D} (f(w(x)) - \mathbb 1(1=y)) \frac{\partial}{\partial w_j} w(x) 
\end{eqnarray}$$