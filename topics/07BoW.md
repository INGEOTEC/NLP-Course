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

Before solving the log-likelihood, it is essential to relate this concept with cross-entropy. First the expectation of $$h(\mathcal X)$$ is computed as $$\sum_{k \in \mathcal X} h(k) f(k)$$ where $$f$$ is the mass function, this can be expressed as $$\mathbb E_f[h(\mathcal X)]$$. On the other hand, the information content of an event is a decreasing function that has its zero when the event has the highest probability, meaning that there is no information carried on an event that occurs always. The information content can be modeled with the function $$I_f(e) = \log(\frac{1}{f(e)})=-\log(f(e)).$$  The **entropy** measures the expected value of the information content that is $$\mathbb E_f[I_f(\mathcal X)]=-\sum_{k \in \mathcal X} f(k) \log(f(k)).$$ Finally, the **cross-entropy** between distribution $$p$$ and $$q$$ is defined as $$H(p, q) = \mathbb E_p[I_q(\mathcal X)] = -\sum_{k \in \mathcal X} p(k) \log(q(k)).$$ 

It can be observed that the negative of the log-likelihood is accumulating the cross-entropy for all the elements in the dataset $$\mathcal D$$, i.e., probability $$p(k)$$ is $$\mathbb 1(y=k)$$, and $$q(k)=f_k(x)$$ using the previous definition of cross-entropy $$H(p, q)$$. The characteristic to note is that $$y$$ and $$x$$ are constants in the inner summation, and variable $$k$$ goes from all the classes. Therefore, minimizing the log-likelihood is minimizing the cross-entropy, which acts as the loss function $$L$$ in the [optimization problem.](/NLP-Course/topics/02Vocabulary/#eq:supervised-learning-optimization) 

$$\begin{eqnarray}
-\frac{\partial}{\partial w_j}  l_{f_\mathcal Y}(f) &=& \frac{\partial}{\partial w_j} \sum_{(x, y) \in \mathcal D} \overbrace{- \sum_{k=1}^K \mathbb 1(k=y) \log f_k(x)}^{cross-entropy} \\
&=& -\sum_{(x, y) \in \mathcal D} \sum_{k=1}^K  \mathbb 1(k=y) \frac{\partial}{\partial w_j} \log f_k(x)\\
&=& - \sum_{(x, y) \in \mathcal D} \sum_{k=1}^K  \frac{\mathbb 1(k=y)}{f_k(x)} \frac{\partial}{\partial w_j} f_k(x)
\end{eqnarray}$$

The function $$f_k$$ has a constraint given that it is taking the place of parameter $$\mathbf p$$ of the Categorical Distribution; this restriction is $$\sum_k^K f_k(x) = 1$$ which can be complied by dividing it with a normalization factor as:

$$f_k(x) = \frac{h_k(w_k(x))}{\sum_{\ell=1}^K h_\ell(w_\ell(x))}$$

The next step is to compute the partial derivative with respect to $$w_j$$ 

$$\begin{eqnarray}
\frac{\partial}{\partial w_j} f_k(x) &=& \frac{\partial}{\partial w_j} \frac{h_k(w_k(x))}{\sum_{\ell=1}^K h_\ell(w_\ell(x))}\\
&=& \frac{\sum_{\ell=1}^K h_\ell(w_\ell(x)) \frac{\partial}{\partial w_j} h_k(w_k(x)) - h_k(w_k(x)) \frac{\partial}{\partial w_j} \sum_{\ell=1}^K h_\ell(w_\ell(x))}{(\sum_{\ell=1}^K h_\ell(w_\ell(x)))^2}\\
&=& \frac{\sum_{\ell=1}^K h_\ell(w_\ell(x)) \frac{\partial}{\partial w_j} h_k(w_k(x)) - h_k(w_k(x)) \frac{\partial}{\partial w_j} h_j(w_j(x))}{(\sum_{\ell=1}^K h_\ell(w_\ell(x)))^2}.
\end{eqnarray}$$

Substituting the partial derivative of $$f_k$$ into the negative log-likelihood is obtained  

$$\begin{eqnarray}
-\frac{\partial}{\partial w_j}  l_{f_\mathcal Y}(f) &=& - \sum_{(x, y) \in \mathcal D} \sum_{k=1}^K  \frac{\mathbb 1(k=y)}{f_k(x)} \frac{\partial}{\partial w_j} f_k(x)\\
&=& - \sum_{(x, y) \in \mathcal D} \sum_{k=1}^K \frac{\mathbb 1(k=y)}{h_k(w_k(x))}   \frac{\sum_{\ell=1}^K h_\ell(w_\ell(x)) \frac{\partial}{\partial w_j} h_k(w_k(x)) - h_k(w_k(x)) \frac{\partial}{\partial w_j} h_j(w_j(x))}{\sum_{\ell=1}^K h_\ell(w_\ell(x))}
\end{eqnarray}$$

# Multinomial Logistic Regression

In order to progress with derivation, it is needed to make some assumptions; the assumption that produce the Multinomial Logistic Regression algorithm is that $$h_k$$ is the exponent, i.e., $$h_k(x)= \exp(x)$$ and the resulting $$f_k$$ is the softmax function. 


$$\begin{eqnarray}
h_k(w_k(x)) &=& \exp(w_k(x))\\
f_k(w_k(x)) &=& \frac{\exp(w_k(x))}{\sum_\ell \exp(w_\ell(x))}
\end{eqnarray}$$

Using $$f_k$$ as the softmax function in the negative log-likelihood produces the following 

$$\begin{eqnarray}
-\frac{\partial}{\partial w_j}  l_{f_\mathcal Y}(f) &=&  - \sum_{(x, y) \in \mathcal D} \sum_{k=1}^K  \frac{\mathbb 1(k=y)}{\exp(w_k(x))} \frac{\sum_{\ell=1}^K \exp(w_\ell(x)) \frac{\partial}{\partial w_j} \exp(w_k(x)) - \exp(w_k(x)) \exp(w_j(x)) \frac{\partial}{\partial w_j} w_j(x)}{\sum_{\ell=1}^K \exp(w_\ell(x))}\\
&=& - \sum_{(x, y) \in \mathcal D} \sum_{k=1}^K  \frac{\mathbb 1(k=y)}{\exp(w_k(x))} \frac{\sum_{\ell=1}^K \exp(w_\ell(x)) \exp(w_k(x)) \frac{\partial}{\partial w_j} w_k(x) - \exp(w_k(x)) \exp(w_j(x)) \frac{\partial}{\partial w_j} w_j(x)}{\sum_{\ell=1}^K \exp(w_\ell(x))}\\
&=& - \sum_{(x, y) \in \mathcal D} \sum_{k=1}^K  \mathbb 1(k=y) \frac{\sum_{\ell=1}^K \exp(w_\ell(x))  \frac{\partial}{\partial w_j} w_k(x) - \exp(w_j(x)) \frac{\partial}{\partial w_j} w_j(x)}{\sum_{\ell=1}^K \exp(w_\ell(x))}\\
&=& - \sum_{(x, y) \in \mathcal D} \sum_{k=1}^K  \mathbb 1(k=y) \left[ \frac{\partial}{\partial w_j} w_k(x) - f_j(w_j(x)) \frac{\partial}{\partial w_j} w_j(x)\right]\\
&=& - \sum_{(x, y) \in \mathcal D}  \mathbb 1(j=y) \left[ \frac{\partial}{\partial w_j} w_j(x) - f_j(w_j(x)) \frac{\partial}{\partial w_j} w_j(x)\right] + \sum_{k \neq j}^K  \mathbb 1(k=y) \left[ \frac{\partial}{\partial w_j} w_k(x) - f_j(w_j(x)) \frac{\partial}{\partial w_j} w_j(x)\right]\\
&=& - \sum_{(x, y) \in \mathcal D}  \mathbb 1(j=y) \left[ 1 - f_j(w_j(x)) \right] \frac{\partial}{\partial w_j} w_j(x) + \sum_{k \neq j}^K  -\mathbb 1(k=y) f_j(w_j(x)) \frac{\partial}{\partial w_j} w_j(x)\\
&=& - \sum_{(x, y) \in \mathcal D}  \left( \mathbb 1(j=y) - f_j(w_j(x)) \right) \frac{\partial}{\partial w_j} w_j(x)\\
&=& \sum_{(x, y) \in \mathcal D}  \left( f_j(w_j(x)) - \mathbb 1(j=y) \right) \frac{\partial}{\partial w_j} w_j(x).
\end{eqnarray}$$

# Logistic Regression

On the other hand, the Logistic Regression algorithm is obtained when one assumes that $$f_1$$ is the sigmoid function and there are two classes; furthermore, this assumption makes it possible to express $$f_2$$ in terms of $$f_1$$ as follows:

$$\begin{eqnarray}
f(x) &=& \frac{1}{1 + \exp(-x)}\\
f_1(x) &=& f(w(x)) \\
f_2(x) &=& 1 - f_1(x).
\end{eqnarray}$$

Using $$f_1$$, $$f_2$$, and the sigmoid function in the negative log-likelihood produce the following

$$\begin{eqnarray}
-\frac{\partial}{\partial w_j}  l_{f_\mathcal Y}(f) &=& - \sum_{(x, y) \in \mathcal D} \sum_{k=1}^K  \frac{\mathbb 1(k=y)}{f_k(x)} \frac{\partial}{\partial w_j} f_k(x)\\
&=& - \sum_{(x, y) \in \mathcal D} \frac{\mathbb 1(1=y)}{f_1(x)} \frac{\partial}{\partial w_j} f_1(x) + \frac{\mathbb 1(2=y)}{1 - f_1(x)} \frac{\partial}{\partial w_j} \left(  1 - f_1(x) \right)\\
&=& - \sum_{(x, y) \in \mathcal D} \frac{\mathbb 1(1=y)}{f(w(x))} \frac{\partial}{\partial w_j} f(w(x)) - \frac{\mathbb 1(2=y)}{1 - f(w(x))} \frac{\partial}{\partial w_j} f(w(x))\\
&=& - \sum_{(x, y) \in \mathcal D} \left[ \frac{\mathbb 1(1=y)}{f(w(x))} - \frac{\mathbb 1(2=y)}{1 - f(w(x))} \right] \frac{\partial}{\partial w_j} f(w(x))\\
&=& - \sum_{(x, y) \in \mathcal D} \left[ \frac{\mathbb 1(1=y)}{f(w(x))} - \frac{\mathbb 1(2=y)}{1 - f(w(x))} \right] (1 - f(w(x))) f(w(x)) \frac{\partial}{\partial w_j} w(x)\\
&=& - \sum_{(x, y) \in \mathcal D} \left[ (1 - f(w(x)))\mathbb 1(1=y) - f(w(x))\mathbb 1(2=y) \right] \frac{\partial}{\partial w_j} w(x)\\
&=& - \sum_{(x, y) \in \mathcal D} (\mathbb 1(1=y) - f(w(x))) \frac{\partial}{\partial w_j} w(x)\\
&=& \sum_{(x, y) \in \mathcal D} (f(w(x)) - \mathbb 1(1=y)) \frac{\partial}{\partial w_j} w(x) 
\end{eqnarray}.$$

It can be observed that the form of the negative log-likelihood for the Multinomial Logistic Regression and the Logistic Regression is similar; the only difference is that there is a function $$w$$ for each class in the multinomial case, and only one function for the Logistic Regression. 

# Text Categorization

Additionally, there has been no assumption regarding the form of $$w(x)$$; given that the problem is text categorization, the variable $$x$$ corresponds to a text. However, the standard definition of Multinomial Logistic Regression and Logistic Regression is that function $$w$$ is a linear function, i.e., $$w(x) = \mathbf w \cdot \mathbf x + w_0$$ where $$\mathbf w \in \mathbb R^d$$, $$\mathbf x \in \mathbb R^d$$, and $$w_0 \in \mathbb R$$. Consequently, one needs to define a function $$m: text \rightarrow \mathbb R^d$$ so that $$m(x) \in \mathbb R^d$$; the Multinomial Logistic Regression in this problem turns out to be:

$$\mathbb P(\mathcal Y=k \mid \mathcal X=x) = \frac{\exp(\mathbf w_k m(x) + w_{k_0})}{\sum_{j=1}^K \exp(\mathbf w_j m(x) + w_{k_0})}.$$

The denominator in the previous equation acts as a normalization factor, and the predicted class is invariant to this normalization factor. Additionally, the logarithm of $$\mathbb P(\mathcal Y=k \mid \mathcal X=x)$$ does not affect the prediction class with the rule $$\textsf{class(x)} = \textsf{arg max}_k \mathbb P(\mathcal Y=k \mid \mathcal X=x).$$ Considering these factors the class is predicted as 

$$\textsf{class(x)} = \textsf{arg max}_k \mathbf w_k m(x) + w_{k_0}.$$

The [first approach](/NLP-Course/topics/06TextCategorization/#sec:tc-categorical) followed to tackle the problem of Text Categorization was to use the Bayes' Theorem $$(\mathbb P(\mathcal Y \mid \mathcal X) = \frac{\mathbb P(\mathcal X \mid \mathcal Y) \mathbb P(\mathcal Y)}{\mathbb P(\mathcal X)})$$ where the likelihood ($$\mathbb P(\mathcal X \mid \mathcal Y)$$) is assumed to be a Categorical Distribution. A Categorical Distribution is defined with a vector $$\mathbf p \in \mathbb R^d$$ where $$d$$ is the different distribution outcomes. The likelihood is a Categorical Distribution given the class $$\mathcal Y$$, therefore there is a parameter $$\mathbf p$$ for each class, which can be identified with a subindex $$k$$, e.g., $$\mathbf p_k$$ is the parameter corresponding to the class $$k$$. The parameters are estimated assumming indepedence, i.e., $$\mathbb P(\mathcal X=w_1,w_2,\ldots,w_\ell \mid \mathcal Y) = \prod_i^\ell \mathbb P(w_i \mid \mathcal Y),$$ where $$w_i$$ is the $$i$$-th token in the text.

The evidence $$\mathbb P(\mathcal X)$$ is a normalization factor in Bayes' theorem, so it does not affect the predicted class; moreover, the logarithm does not change the prediction value. Incorporating into Bayes' theorem these transformations, it is obtained

$$\log \mathbb P(\mathcal Y \mid \mathcal X=w_1,w_2,\ldots,w_\ell) \propto \sum_i^\ell \log \mathbb P(w_i \mid \mathcal Y) + \log P(\mathcal Y);$$

which can be expressed using the parameter, $$\mathbf p_k$$, of the Categorical Distribution, and the frequency of each token as follows:

$$\log \mathbb P(\mathcal Y=k \mid \mathcal X=x) \propto \sum_i^d \log(\mathbf p_{k_i}) \textsf{freq}_i(x) + \log P(\mathcal Y),$$

where $$\textsf{freq}_i(x)$$ computes the frequency of the token identified with the index $$i$$ in the text $$x$$. In order to illustrate the similarity between the previous equation and the one obtained with Multinomial Logistic Regression, it is convenient to express it using vectors, i.e., $$\log \mathbb P(\mathcal Y=k \mid \mathcal X=x) \propto \log(\mathbf p_k) \textsf{freq}(x) +  \log P(\mathcal Y),$$ where $$\log(\mathbf p_k) \in \mathbb R^d$$, $$\textsf{freq}(x) \in \mathbb R^d$$, and $$\log \mathbb P(\mathcal Y=k) \in \mathbb R.$$ Therefore, the parameters $$\log(\mathbf p_k)$$ and $$\log \mathbb P(\mathcal Y=k)$$ are equivalent to $$\mathbf w$$ and $$w_0$$ in the (Multinomial) Logistic Regression, and $$m(x)$$ is the frequency, $$\textsf{freq}(x)$$, in the Categorical Distribution approach.
 
# Gradient Descent Algorithm

The parameters $$\mathbf w$$ and $$w_0$$ can be estimated by minimizing the negative log-likelihood or equivalently by using the cross-entropy as loss function. Unfortunately, the system of equations $$-\frac{\partial}{\partial w_j} l_{f_\mathcal Y}(f) = 0$$ cannot be solved analytically, so one needs to rely on numerical methods to find the $$w_j$$ value that makes the function minimal. One approach that has been very popular lately is the gradient descent algorithm. The idea is that the parameter $$w_j$$ can be found iterative using the update rule

$$w^i_j = w^{i-1}_j - \eta \frac{\partial}{\partial w_j} \sum_{(x, y) \in \mathcal D} L(y, g(x)).$$

The Logistic Regression update rule is

$$w^i_j = w^{i-1}_j - \eta \sum_{(x, y) \in \mathcal D} (f(\mathbf w \cdot \mathbf x + w_0) - \mathbb 1(1=y)) \frac{\partial}{\partial w_j} \mathbf w \cdot \mathbf x + w_0,$$

and the Multinomial Logistic Regression corresponds to

$$\mathbf w^i_{j_\ell} = \mathbf w^{i-1}_{j_\ell} - \eta \sum_{(x, y) \in \mathcal D}  \left( f_j(\mathbf w_j \cdot \mathbf x + w_0) - \mathbb 1(j=y) \right) \frac{\partial}{\partial \mathbf w_{j_\ell}} \mathbf w_j \cdot \mathbf x + w_{j_0}.$$

# Term Frequency




