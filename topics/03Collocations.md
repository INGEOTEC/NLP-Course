---
layout: default
title: Collocations
nav_order: 3
---

# Collocations
{: .fs-10 .no_toc }

## Contents
{: .no_toc .text-delta }

1. TOC
{:toc}

## Libraries used
{: .no_toc .text-delta }
```python
from text_models import Vocabulary
```

## Installing external libraries
{: .no_toc .text-delta }

```bash
pip install microtc
pip install evomsa
pip install text_models
```

---

# Introduction

A collocation is an expression with the characteristic that its meaning cannot be inferred by simply adding the definition of each of its words, e.g., kick the bucket.

As can be expected, finding a collocation is a challenging task; one needs to know the meaning of the words and then realize that the combination of these words produces an utterance that its components cannot explain. However, the approach taken here is more limited; instead of looking for a sentence of any length, we will analyze bi-grams and describe algorithms that can retrieve bi-grams that could be considered collocations. 

The frequency of the bigrams can be represented in a co-occurrence matrix as the one shown in the following table. 

|    | the     | to      | of      | in      | and     | 
|----|---------|---------|---------|---------|---------|    
|the |       0 |  453796 |  435030 |  323450 |  317663 |
|to  |  453796 |       0 |  170941 |  161891 |  228785 |
|of  |  435030 |  170941 |       0 |  122502 |  130662 |
|in  |  323450 |  161891 |  122502 |       0 |  125997 |
|and |  317663 |  228785 |  130662 |  125997 |       0 |

Table: Co-occurence matrix
{: #co-occurrence .label }

The co-occurrence matrix was created using the data obtained from the library `text_models` using the following code. The third line retrieves all the bigrams and stores them in the variable `bigrams.` The loop goes for all the bigrams that contains one of the five words defined in `index,` it is observed that the matrix is symmetric, this is because the `text_models` library does not store the order of the words composing the bigram.

```python 
date = dict(year=2022, month=1, day=10)
voc = Vocabulary(date, lang='En')
bigrams = Counter({k: v for k, v in voc.voc.items() if k.count("~")})
co_occurrence = np.zeros((5, 5))
index = {'the': 0, 'to': 1, 'of': 2, 'in': 3, 'and': 4}

for bigram, cnt in bigrams.most_common():
    a, b = bigram.split('~')
    if a in index and b in index:
        co_occurrence[index[a], index[b]] = cnt
        co_occurrence[index[b], index[a]] = cnt
```

The idea is to use the information of the co-occurrence matrix to find the pairs of words that can be considered collocations. The first step is to transform the co-occurrence matrix into a bivariate distribution and then use statistical approaches to retrieve some prominent pairs. Before going into the details of these algorithms, it is pertinent to describe the relationship between words and random variables.

Each element in the matrix can be uniquely identified by the pair words, e.g., the frequency of pair (_in_, _of_) is $$122502$$. However, it is also possible to identify the same element using an index. For example, if the first word (_the_) is assigned the index $$0$$, the index $$3$$ corresponds to word _in_ and $$2$$ to _of_. Consequently, the element (_in_, _of_) can uniquely identify with the pair (3, 2). One can create a mapping between words and natural numbers such that each different word has a unique identifier. The mapping allows working with natural numbers instead of words which facilitates the analysis and returns to the words (using the inverse mapping) when the result is obtained. 

The mapping can be implemented using a dictionary, as seen from the following code where the variable of interest is `index.` 

```python
for bigram, cnt in bigrams.items():
    a, b = bigram.split('~')
    for x in [a, b]:
        if x not in index:
            index[x] = len(index)
len(index)
9598
```

It is essential to mention that a random variable is a mapping that assigns a real number to each outcome. In this case, the outcome is observing a word and the mapping is the transformation of the word into the natural number. 

The co-occurrence matrix contains the information of two random variables; each one can have $$d$$ (length of the dictionary) different outcomes. Sometimes, working with two random variables might be challenging, so a more suitable approach is starting the description with the most simple case, which corresponds to a single random variable with only two outcomes. 

# Bernoulli Distribution

Let $$\mathcal{X}$$ be a random variable with two outcomes ($$\{1, 0\}$$), e.g., this variable corresponds to a language that only has two words. At this point, one might realize that different experiments can be represented with a random variable of two outcomes; perhaps the most famous one is tossing a coin.

The random variable $$\mathcal{X}$$ has a Bernoulli distribution, i.e., $$\mathcal X \sim \textsf{Bernoulli}(p)$$, in the case that $$\mathbb P(\mathcal X=1)=p$$ and $$\mathbb P(\mathcal X=0)=1 - p$$ for $$p \in [0, 1]$$, where the probability (mass) function is $$f_{\mathcal X}(x) = p^x(1-p)^{1-x}.$$

For example, the language under study has two words _good_ and _bad_, and we encountered a sequence "good bad bad good good." Using the following mapping 

$$\mathcal X(w) = \begin{cases}
1 \text{when } w \text{ is good}\\
0 \text{ when } w \text{ is good}
\end{cases}.$$

The sequence is represented as $$(1, 0, 0, 1, 1)$$ and in general a sequence of five elements is $$(\mathcal X_1, \mathcal X_2, \mathcal X_3, \mathcal X_4, \mathcal X_5);$$ as expected a sequence of $$N$$ observations is $$(\mathcal X_1, \mathcal X_2, \ldots, \mathcal X_N)$$. Different studies can be applied to a sequence; however, at this point we would like to impose some constraints on the way it was obtained. The first constraint is to assume that $$\mathcal X_i \sim \textsf{Bernoulli}(p),$$ then one is interested in estimating the value of the parameter $$p$$. The second assumption is that the random variables are indepedent, that is, the outcome of the variable $$\mathcal X_i$$ is independent of $$\mathcal X_j$$ for $$j \neq i.$$

There is an important characteristic for independen random variables, i.e., $$(\mathcal X_1, \mathcal X_2, \ldots, \mathcal X_N)$$ which is 

$$\mathbb P(\mathcal X_1, \mathcal X_2, \ldots, \mathcal X_N) = \prod_{i=1}^N \mathbb P(\mathcal X_i).$$

Returning to the example "good bad bad good good," the independent assumption means that observing _bad_ as the second word is not influenced by getting _good_ as the first word. 

## Maximum Likelihood Method

A natural example of independent random variables is the tossing of a coin. For example, observing the sequence $$(1, 0, 0, 1, 1)$$ and knowing that these come from tossing a coin five times, then our intuition indicates that the estimated parameter $$p$$ correspond to the fraction between the number of ones (3) and the number of tosses (5). 

A natural example of independent random variables is the tossing of a coin. For example, observing the sequence $$(1, 0, 0, 1, 1)$$ and knowing that these come from tossing a coin five times, then our intuition indicates that the estimated parameter $$p$$ correspond to the fraction between the number of ones (3) and the number of tosses (5). In this case, our intuition corresponds to the maximum likelihood method defined as follows:

$$\mathcal L_{f_{\mathcal X}}(\theta) = \prod_{i=1}^N f_{\mathcal X}(x_i \mid \theta),$$

where $$f_{\mathcal X}$$ corresponds to the probability density function, in the case discrete random variable corresponds to $$\mathbb P(X=x) = f_{\mathcal X}(x),$$ and the notation $$f_{\mathcal X}(x_i \mid \theta)$$ indicates that $$f$$ depends on a set of parameters refered as $$\theta$$.

The maximum likelihood estimator $$\hat \theta$$ corresponds to maximizing $$\mathcal L_{f_{\mathcal X}}(\theta)$$ or equivalent maximizing $$l_{f_\mathcal X}(\theta) =  \log \mathcal L_{f_{\mathcal X}}(\theta).$$ 

Continuing with the example $$(1, 0, 0, 1, 1)$$, given that $$\mathcal X_i$$ is Bernoulli distributed then $$f_{\mathcal X}(p) = p^x(1-p)^{1-x}$$. The maximum likelihood estimator of $$p$$ is obtained by maximizing the likelihood function, which can be solved analytically by following the next steps.

$$\begin{eqnarray} 
\frac{d}{dp} \mathcal l_{f_{\mathcal X}}(p) &=& 0 \\ 
\frac{d}{dp} [ \sum_{i=1}^N x_i \log p + (1-x_i) \log (1 - p)] &=& 0 \\ 
\frac{d}{d p} [ \sum_{i=1}^N x_i \log p + \log (1 - p) (N - \sum_{i=1}^N x_i) ] &=& 0\\ 
\sum_{i=1}^N x_i \frac{d}{d p} \log \mathcal p + (N - \sum_{i=1}^N x_i) \frac{d}{d p} \log (1 - \mathcal p) &=& 0\\
\sum_{i=1}^N x_i \frac{1}{p} + (N - \sum_{i=1}^N x_i) \frac{-1}{(1 - p)} &=& 0 \\
\end{eqnarray},$$

solving for $$p$$ it is obtained $$\hat p = \frac{1}{N}\sum_{i=1}^N x_i.$$

For example, the following code creates an array of 100 elements where each element is the outcome of a Bernoulli distributed random variable. The second line estimates the parameter $$p$$. 

```python
x = np.random.binomial(1, 0.3, size=100)
hp = x.mean()
```

# Categorical distribution

Having a language with only two words seems useless; it sounds more realistic to have a language with $$d$$ words. Let $$\mathcal X$$ be a random variable with $$d$$ outcomes ($$\{1, 2, \ldots, d\}$$). The random variable $$\mathcal X$$ has a Categorical distribution, i.e., $$\mathcal X \sim \textsf{Categorical}(\mathbf p)$$ in the case $$\mathbb P(\mathcal X=i) = \mathbf p_i $$ for $$1 \leq i \leq d$$, where $$\sum_i^d \mathbf p_i =1$$ and $$\mathbf p \in \mathbb R^d$$. The probability mass function of a Categorical distribution is $$f_{\mathcal X}(x) = \prod_{i=1}^d \mathbf p_i^{\delta(i=x)}$$.

The estimated parameter is $$\hat{\mathbf p}_i = \frac{1}{N}\sum_{j=1}^N \delta(x_j = i).$$

## Maximum Likelihood Estimator

The maximum likelihood estimator can be obtained by maximizing the log-likelihood, i.e., 

$$l_{f_\mathcal X}(\mathbf p_j) = \log \prod_{i=1}^N \prod_{k=1}^d \mathbf p_k^{\delta(x_i=k)},$$ 

subject to the constraint $$\sum_i^d \mathbf p_i=1$$. An optimization problem with a equality constraint can be solved using Langrage multipliers which requieres setting the constraint in the original formulation and making a derivative on a introduce variable $$\lambda$$. Using Langrage multiplier the system of equations that need to be solved is the following: 

$$\begin{eqnarray}
\frac{\partial}{\partial \mathbf p_j} [\log \prod_{i=1}^N \prod_{k=1}^d \mathbf p_k^{\delta(x_i=k)} - \lambda (\sum_i^d \mathbf p_i -1)] &=& 0 \\
\frac{\partial}{\partial \lambda} [\log \prod_{i=1}^N \prod_{k=1}^d \mathbf p_k^{\delta(x_i=k)} - \lambda (\sum_i^d \mathbf p_i -1)] &=& 0 \\
\end{eqnarray},$$

where the term $$\lambda (\sum_i^d \mathbf p_i -1)$$

For example, one of the most known processes that involve a Categorical distribution is rolling a dice. The following is a procedure to simulate dice rolling using a Multinomial distribution. 

```python
X = np.random.multinomial(1, [1/6] * 6, size=100)
x = X.argmax(axis=1)
```

On the other hand, the maximum likelihood estimator can be implemented as follows:

```python
var, counts = np.unique(x, return_counts=True)
N = counts.sum()
p = counts / N
```

# Bivariate distribution

We have all the elements to realize that the co-occurrence matrix is the realization of two random variables (each one can have $$d$$ outcomes); it keeps track of the number of times a bigram appear in a corpus. So far, we have not worked with two random variables; however, the good news is that the co-occurrence matrix contains all the information needed to define a bivariate distribution for this process.

The first step is to define a new random variable $$\mathcal X_i$$ that represents the event of getting a bigram. $$\mathcal X_i$$ is defined using $$\mathcal X_r$$ and $$\mathcal X_r$$  which correspond to the random variables of the first and second word of the bigram. For the case, $$\mathcal X_r=r$$ and $$\mathcal X_c=c$$ the random variable is defined as $$\mathcal X_i= (r + 1) \cdot (c + 1)$$, where the constant one is needed in case zero is included as one of the outcomes of the random variables $$\mathcal X_r$$ and $$\mathcal X_c$$. For example, in the co-occurrence matrix, presented previously, the realization $$\mathcal X_r=3$$ and $$\mathcal X_c=2$$ corresponds to the bigram (_in_, _of_) which has a recorded frequency of $$122502$$, using the new variable the event is $$\mathcal X_i=12$$. As can be seen, $$X_i$$ is a random variable with $$d^2$$ outcomes. We can suppose $$\mathcal X_i$$ is a Categorical distributed random variable, i.e., $$\mathcal X_i \sim \textsf{Categorical}(\mathbf p).$$

The fact that $$\mathcal X_i$$ would be considered Categorical distributed implies that the bigrams are independent that is observing one of them is not affected by the previous words in any way. However, with this assumption it is straightforward estimating the parameter $$\mathbf p$$ which is $$\hat{\mathbf p}_i = \frac{1}{N}\sum_{j=1}^N \delta(x_j = i).$$. Consequently, the co-occurrence matrix can be converted into a bivariate distribution by dividing it by $$N$$, where $$N$$ is the sum of all the values of the co-occurrence matrix. For example, the following code builds the bivariate distribution from the co-occurrence matrix.

```python
co_occurrence = np.zeros((len(index), len(index)))
for bigram, cnt in bigrams.most_common():
    a, b = bigram.split('~')
    if a in index and b in index:
        co_occurrence[index[a], index[b]] = cnt
        co_occurrence[index[b], index[a]] = cnt
co_occurrence = co_occurrence / co_occurrence.sum()
```

The following table presents an extract of the bivariate distribution. 

|    | the      | to       | of       | in       | and      | 
|----|----------|----------|----------|----------|----------|    
|the |  0.00000 |  0.00120 |  0.00115 |  0.00086 |  0.00084 |
|to  |  0.00120 |  0.00000 |  0.00045 |  0.00043 |  0.00061 |
|of  |  0.00115 |  0.00045 |  0.00000 |  0.00033 |  0.00035 |
|in  |  0.00086 |  0.00043 |  0.00033 |  0.00000 |  0.00033 |
|and |  0.00084 |  0.00061 |  0.00035 |  0.00033 |  0.00000 |

Once the information of the bigrams has been transformed into a bivariate distribution, we can start analyzing it. As mentioned previously, the idea is to identify those bigrams that can be considered collocations. However, the frequency of the bigrams does not contain semantic information of the words or the phrase at hand, which can be used to identify a collocation precisely. Nonetheless, a collocation is a phrase where its components do not appear by chance; that is, the elements composing it are not drawn independently from a distribution. Therefore, the bivariate distribution can be used to identify those words that are not independent, which is a hard constraint for being considered a collocation. 

## Independence and Marginal Distribution

The bivariate distribution shown in the previous table contains the probability of obtaining a bigram, i.e., $$\mathbb P(\mathcal X_r=r, \mathcal X_c=c)$$; this information is helpful when combined with the concept of independence and marginal distribution.

Two random variables $$\mathcal X$$ and $$\mathcal Y$$ are **independent** if
$$\mathbb P(\mathcal X, \mathcal Y)=\mathbb P(\mathcal X) \mathbb(\mathcal Y).$$ 

The definition of independence is useless if $$\mathbb P(\mathcal X)$$ and $$\mathbb P(\mathcal Y)$$ are unknown. Fortunately, the **marginal distribution** definition describes the procedure to obtain $$\mathbb P(\mathcal X=x)$$ and $$\mathbb P(\mathcal Y=y)$$ from the bivariate distribution. Let $$f_{\mathcal X, \mathcal Y}$$ be the joint distribution mass function (i.e., $$f_{\mathcal X, \mathcal Y}(x, y)=\mathbb P(\mathcal X=x, \mathcal Y=y)$$) then the marginal mass function for $$\mathcal X$$ is 

$$f_{\mathcal X}(x) = \mathbb P(\mathcal X=x) = \sum_y \mathbb P(\mathcal X=x, \mathcal Y=y) = \sum_y f_{\mathcal X, \mathcal Y}(x, y).$$

# Example of rolling two dices

The interaction of these concepts can be better understood with a simple example in where all details are known. The following code simulated the rolling of two dices. Variables `R` and `C` contain the rolling of the two dices, and the variable `Z` has the outcome of the pair. 

```python
d = 6
R = np.random.multinomial(1, [1/d] * d, size=10000).argmax(axis=1)
C = np.random.multinomial(1, [1/d] * d, size=10000).argmax(axis=1)
Z = [[r, c] for r, c in zip(R, C)]
```

`Z` is transformed first into a frequency matrix (variable `W`), equivalent to a co-occurrence matrix, and then, it is converted into a bivariate distribution (last line).

```python
W = np.zeros((d, d))
for r, c in Z:
    W[r, c] += 1
W = W / W.sum()
```

The next matrix presents the bivariate distribution `W`

$$
W=\mathbb P(\mathcal R, \mathcal C) = 
\begin{pmatrix}
0.0295 & 0.0286 & 0.0261 & 0.0265 & 0.0272 & 0.0313 \\
0.0280 & 0.0293 & 0.0299 & 0.0273 & 0.0280 & 0.0272 \\
0.0261 & 0.0266 & 0.0258 & 0.0268 & 0.0299 & 0.0294 \\
0.0301 & 0.0287 & 0.0274 & 0.0290 & 0.0254 & 0.0284 \\
0.0263 & 0.0279 & 0.0276 & 0.0287 & 0.0301 & 0.0273 \\
0.0270 & 0.0270 & 0.0254 & 0.0280 & 0.0241 & 0.0281 \\
\end{pmatrix}.$$

The next step is to compute the marginal instributions $$\mathbb P(\mathcal R)$$ and $$\mathbb P(\mathcal C)$$ which can be done as follows

```python
R_m = W.sum(axis=1)
C_m = W.sum(axis=0)
```

The marginal distribution and the definition of independence are used to obtain $$\mathbb P(\mathcal R) \mathbb P(\mathcal C)$$ which can be computed using the dot product, consequenly `R` and `C` need to be transform into a two dimensional
array using `np.atleast_2d` function. 

```python
ind = np.dot(np.atleast_2d(R).T, np.atleast_2d(C))
```

`W` contains the estimated bivariate distribution, and `ind` could be the bivariate distribution only if $$\mathcal R$$ and $$\mathcal C$$ are independent. The following matrix shows `W-ind`, which would be a zero matrix if the variables are independent. 

$$\mathbb P(\mathcal R, \mathcal C) - \mathbb P(\mathcal R)\mathbb P(\mathcal C) = 
\begin{pmatrix}
0.0012 & 0.0002 & -0.0013 & -0.0016 & -0.0007 & 0.0022 \\
-0.0003 & 0.0008 & 0.0024 & -0.0009 & 0.0001 & -0.0019 \\
-0.0014 & -0.0011 & -0.0009 & -0.0006 & 0.0028 & 0.0011 \\
0.0019 & 0.0003 & -0.0000 & 0.0009 & -0.0024 & -0.0006 \\
-0.0017 & -0.0003 & 0.0004 & 0.0008 & 0.0024 & -0.0015 \\
0.0003 & 0.0002 & -0.0005 & 0.0015 & -0.0022 & 0.0007 \\
\end{pmatrix}
$$

It is observed from the matrix that all its elements are close to zero ($$\mid W_{ij}\mid \leq 0.005$$), which is expected given that by construction, the two variables are independent. On the other hand, a simulation where the variables are not independent would produce a matrix where its components are different from zero. Such an example can be quickly be done by changing variable `Z.` The following code simulates the case where the two dices cannot have the same value: the events $$(1, 1), (2, 2), \ldots,$$ are unfeasible. It is hard to imagine how this experiment can be done with two physical dice; however, simulating it is only a condition, as seen in the following code.   

```python
Z = [[r, c] for r, c in zip(R, C) if r != c]
```

The difference between the estimated bivariate distribution and the product of the marginal distributions is presented in the following matrix. It can be observed that the values in the diagonal are negative because $$\mathbb P(\mathcal R=x, \mathcal C=x)=0$$ given that the event is not possible in this experiment. 

$$ 
\begin{pmatrix}
-0.0280 & 0.0063 & 0.0037 & 0.0040 & 0.0054 & 0.0085 \\
0.0057 & -0.0284 & 0.0082 & 0.0049 & 0.0063 & 0.0034 \\
0.0037 & 0.0040 & -0.0276 & 0.0046 & 0.0089 & 0.0064 \\
0.0083 & 0.0063 & 0.0052 & -0.0280 & 0.0032 & 0.0050 \\
0.0041 & 0.0058 & 0.0059 & 0.0071 & -0.0270 & 0.0041 \\
0.0062 & 0.0060 & 0.0045 & 0.0075 & 0.0033 & -0.0275 \\
\end{pmatrix}
$$

```python
Z = [[2 if c == 1 and np.random.rand() < 0.1 else r, c] for r, c in zip(R, C)]
```

$$
\begin{pmatrix}
0.0019 & -0.0032 & -0.0007 & -0.0010 & -0.0000 & 0.0029 \\
0.0001 & -0.0012 & 0.0028 & -0.0005 & 0.0004 & -0.0015 \\
-0.0036 & 0.0100 & -0.0031 & -0.0028 & 0.0006 & -0.0011 \\
0.0023 & -0.0017 & 0.0004 & 0.0013 & -0.0020 & -0.0002 \\
-0.0013 & -0.0024 & 0.0008 & 0.0012 & 0.0029 & -0.0011 \\
0.0007 & -0.0015 & -0.0002 & 0.0018 & -0.0019 & 0.0010 \\
\end{pmatrix}
$$


