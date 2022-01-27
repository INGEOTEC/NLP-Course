---
layout: default
title: N-Gram Language Model
nav_order: 4
---

# N-Gram Language Model
{: .fs-10 .no_toc }

## Contents
{: .no_toc .text-delta }

1. TOC
{:toc}

## Libraries used
{: .no_toc .text-delta }
```python
from text_models import Vocabulary
from collections import Counter
import numpy as np
from scipy.stats import norm, chi2
from wordcloud import WordCloud as WC
from matplotlib import pylab as plt
from collections import defaultdict
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

A Language Model (LM) assigns probabilities to words (tokens), sentences, or documents. The direct usage of an LM is to estimate the probability of observing a particular text, it can also be used to generate texts, and in general, it models the dynamics of a language. It is essential to mention that most of the **Natural Language Understanding** (NLU) tasks rely on an LM.

LM deals with modeling the multivariate probability $$\mathbb P(\mathcal X_1, \mathcal X_2, \ldots, \mathcal X_\ell)$$ of observing $$\ell$$ words (tokens).  As expected, these $$\ell$$ random variables are dependent (see the definition of the complement concept, i.e., [independence](/NLP-Course/topics/03Collocations/#sec:independence-marginal).), and in order to work with them, it is needed to define the concept of conditional probability. 
  
# Conditional Probability

The conditional probability of two random variables $$\mathcal X$$ and $$\mathcal Y$$ is defined as:

$$\mathbb P(\mathcal Y \mid \mathcal X) = \frac{\mathbb P(\mathcal X, \mathcal Y)}{\mathbb P(\mathcal X)},$$

if $$\mathbb P(\mathcal Y) > 0.$$

The definition allows defining $$\mathbb P(\mathcal X, \mathcal Y) = \mathbb P(\mathcal Y \mid \mathcal X) \mathbb P(\mathcal X),$$ which is helpful for two words text. The general case involves $$\ell$$ words, which can be defined using the probability chain rule.

$$\begin{eqnarray}
\mathbb P(\mathcal X_1, \ldots, \mathcal X_\ell) &=& \mathbb P(\mathcal X_\ell \mid \mathcal X_1, \ldots, \mathcal X_{\ell -1}) \mathbb P(\mathcal X_1, \ldots, \mathcal X_{\ell - 1})\\ 
\mathbb P(\mathcal X_1, \ldots, \mathcal X_{\ell - 1}) &=& \mathbb P(\mathcal X_{\ell - 1} \mid \mathcal X_1, \ldots, \mathcal X_{\ell - 2}) \mathbb P(\mathcal X_1, \ldots, \mathcal X_{\ell - 2}) \\
&\vdots& \\
\mathbb P(\mathcal X_1, \mathcal X_2) &=& \mathbb P(\mathcal X_2 \mid \mathcal X_1) \mathbb P(\mathcal X_1)
\end{eqnarray}.$$

The first equality of the previous system of equation shows an exciting characteristic; it computes the probability of the next word, $$\ell$$, given a history of $$\ell -1 $$ words; that is

$$\mathbb P(\mathcal X_\ell \mid \mathcal X_1, \ldots, \mathcal X_{\ell -1}) = \frac{\mathbb P(\mathcal X_1, \ldots, \mathcal X_\ell)}{\mathbb P(\mathcal X_1, \ldots, \mathcal X_{\ell - 1})},$$ 

where $$\mathbb P(\mathcal X_1, \ldots, \mathcal X_{\ell - 1}) = \sum_x \mathbb P(\mathcal X_1, \ldots, \mathcal X_{\ell - 1}, \mathcal X_\ell = x)$$ is the marginal distribution. 

# N-Gram Language Model

Traditionally, the multivariate probability distributions are fixed with respect to the number of variables; however, the number of words in a sentence or a document is variable. Nonetheless, the dependency between the latest word and the first one on a long text is negligible. 

Even when the number of variables is constant, that is, $$\ell$$ is kept constant in the model, which has as a consequence that it is not possible to represent a text longer than $$\ell$$ words. There is still another concern that comes from estimating the parameters of the multivariate distribution. $$\mathcal X$$ is Categorical distributed with $$d$$ possible outcomes. The minimum number of examples needed to observe the $$d$$ outcomes is $$d^1$$; in the case of two variables, i.e., $$\ell=2$$, it is needed at least $$d^2$$, and in general, for $$\ell$$ variables it is needed $$d^\ell$$ examples. 

The consequence is that for a relatively small $$\ell$$, one needs a vast dataset to have enough information to estimate the multivariate distribution. We have seen this behavior in the [co-occurrence matrix](/NLP-Course/topics/03Collocations/#tab:co-occurrence) where most matrix elements are zero; only $$0.021$$% of the matrix is different from zero. 

The constraints mentioned above can be handled by approximating the conditional probability of the following word. Instead of using $$\ell - 1$$ words to measure the probability of obtaining word $$\ell$$, the history is fixed to only include the latest $$n$$ words, namely

$$\mathbb P(\mathcal X_\ell \mid \mathcal X_1, \ldots, \mathcal X_{\ell -1}) \approx \mathbb P(\mathcal X_\ell \mid \mathcal X_{\ell - n + 1}, \ldots, \mathcal X_{\ell -1}).$$ 


# Bigrams 

The n-gram LM for $$n-2$$ is known as bigram, and its formulation is $$\mathbb P(\mathcal X_\ell \mid \mathcal X_{\ell-1}).$$ We have been working extensively with bigrams, albeit in a different direction: [finding collocations](/NLP-Course/topics/03Collocations). In addition, the dataset obtained from the library `text_models` does not follow the definition of LM bigrams; an LM bigram uses as input two words (tokens) that are consecutive. Conversely, the two words in `text_models` models did not have an order and were the composition of all the pairs in each tweet. 

The bigram model is defined as:

$$\mathbb P(\mathcal X_\ell \mid \mathcal X_{\ell - 1}) = \frac{\mathbb P(\mathcal X_{\ell -1}, \mathcal X_\ell)}{\mathbb P(\mathcal X_{\ell - 1})}.$$

The procedure to estimate the values of $$\mathbb P(\mathcal X_{\ell -1}, \mathcal X_\ell)$$ and $$\mathbb P(\mathcal X_{\ell - 1})$$ have been described [previously.](/NLP-Course/topics/03Collocations/#sec:bivariate-distribution) The example seen was a pair of dices where a dependency between $$\mathcal X_r=2$$ and $$\mathcal X_c=1$$ is wired. This example can be modified to link it closely to a bigram LM by assuming that there is a language with only four words, represented by $$\{0, 1, 2, 3\}$$. These words are used to compose bigrams following a Categorical distribution, where the process generating this bivariate distribution is the following.

```python
d = 4
R = np.random.multinomial(1, [1/d] * d, size=10000).argmax(axis=1)
C = np.random.multinomial(1, [1/d] * d, size=10000).argmax(axis=1)
rand = np.random.rand
Z = [[r, 2 if r == 1 and rand() < 0.1 else c]
      for r, c in zip(R, C)
     if r != c or (r == c and rand() < 0.2)]
```

It can be observed that the starting point are variables $$\mathcal X_r$$ (i.e., $$\mathcal X_{\ell - 1}$$) and $$\mathcal X_c$$ (i.e., $$\mathcal X_\ell$$) following a categorical distribution with $$\mathbf p_i = \frac{1}{d}$$ where $$d=4$$. Then the bivariate distribution is sampled on variable `Z` where 80% of the examples where $$\mathcal X_r=\mathcal X_c$$ are droped and it is induced a dependency for $$\mathcal X_r=1$$ and $$\mathcal X_c=2$$.   

The estimated $$\mathcal P(\mathcal X_r, \mathcal X_c)$$ is 
{: #bivarite-distribution-bigrams }

$$
\begin{pmatrix}
0.0149 & 0.0771 & 0.0791 & 0.0782 \\
0.0663 & 0.0131 & 0.0939 & 0.0730 \\
0.0810 & 0.0799 & 0.0146 & 0.0796 \\
0.0797 & 0.0742 & 0.0780 & 0.0172 \\
\end{pmatrix}.
$$

The marginal distribution $$\mathbb P(\mathcal X_r) = (0.249374, 0.246370, 0.255133, 0.249124)$$ which can be obtained as follows:

```python
M_r = W.sum(axis=1)
```

where `W` contains the estimated bivariate distribution. It is not necessary to obtain the marginal $$\mathbb P(\mathcal X_c) = (0.241988, 0.244367, 0.265648, 0.247997)$$; however, it is observed that the dependency induce impacts this marginal and not the former.

The conditional $$\mathbb P(\mathcal X_c \mid \mathcal X_r)$$ can be estimated as  

```python
p_l = (W / np.atleast_2d(M_r).T)
```

and the result is shown in the following matrix

$$
\begin{pmatrix}
0.0597 & 0.3092 & 0.3173 & 0.3138 \\
0.2693 & 0.0534 & 0.3811 & 0.2962 \\
0.3175 & 0.3131 & 0.0574 & 0.3121 \\
0.3201 & 0.2980 & 0.3131 & 0.0688 \\
\end{pmatrix}.
$$

## Generating Sequences

The conditional probability $$\mathbb P(\mathcal X_c \mid \mathcal X_r)$$ (variable `p_l`) and the marginal probability $$\mathbb P(\mathcal X_r)$$ (variable `M_r`) can be used to generate a text. The example would be more realistic if we used letters instead of indices; this can be done with a mapping between the index and string, as can be seen below.

```python
id2word = {0: 'a', 1: 'b', 2: 'c', 3: 'd'}
```

It is helpful to define a function (`cat`) that behaves like a Categorical distribution; this can be done using a Multinomial distribution using the parameters shown in the following code.

```python
cat = lambda x: np.random.multinomial(1, x, 1).argmax()
```

The requirement to use the conditional probability is that a starting word is needed; the conditional probability $$\mathbb P(\mathcal X_c \mid \mathcal X_r)$$ once the value is known of $$\mathcal X_r$$. We can assume that the first word can be simulated using the marginal $$\mathbb P(\mathcal X_r)$$ as can be seen as follows.

```python
w1 = cat(M_r)
```

Once the starting word is obtained, it is needed to iterate as many times as one wants using the conditional probability to generate the next token; this can be observed in the following code.  

```python
l = 25
text = [cat(M_r)]
while len(text) < l:
    next = cat(p_l[text[-1]])
    text.append(next)
text = " ".join(map(lambda x: id2word[x], text))
```

The previous code (including the marginal distribution) is executed three times, and the result can be observed in the following table. 

|Text                                             |
|-------------------------------------------------|
|d a d c b d c d c d a b a b c d a d b d a d b c b|
|c a b a c a c d b d d a d c d b d b d b a b a b c|
|b a c b a b a c b d a d c b c d b c d c a c b d c|


## Using a Sequence to Estimate $$\mathbb P(\mathcal X_r, \mathcal X_c)$$

NLP aims to find and estimate the model that can be used to generate text; therefore, it is unrealistic to have $$\mathbb P(\mathcal X_{1}, \mathcal X_{2}, \ldots, \mathcal X_\ell)$$; however, we can estimate it using examples. Considering that we have a method to generate text, we can generate a long sequence and estimate the bivariate distribution parameters from it.

The first step is to have a mapping from words to numbers. The second step is to retrieve the words, and the third step is to create the bigrams. These can be observed in the following code.  

```python
w2id = {v: k for k, v in id2word.items()}
lst = [w2id[x] for x in text.split()]
Z = [[a, b] for a, b in zip(lst, lst[1:])]
```

The rest of the code we have seen previously; however, it is next to facilitate the reading. 

```python
d = len(w2id)
W = np.zeros((d, d))
for r, c in Z:
    W[r, c] += 1
W = W / W.sum()
```

The bivariate distribution estimated from the sequence is presented in the following matrix. It can be observed that the values of this matrix are similar to the [matrix](/NLP-Course/topics/04NGramLM/#bivarite-distribution-bigrams) used to generate the sequence.

$$
\begin{pmatrix}
0.0150 & 0.0750 & 0.0790 & 0.0781 \\
0.0655 & 0.0113 & 0.0930 & 0.0715 \\
0.0869 & 0.0808 & 0.0156 & 0.0810 \\
0.0797 & 0.0743 & 0.0767 & 0.0165 \\
\end{pmatrix}.
$$

## Likelihood of a Sequence

We have all the elements to compute the likelihood of a particular sequence, for example the probability of observing the sequence *a d b c* is $$\mathbb P(\text{"a d b c"}) = 0.008885$$; this is a simplified notation the complete notation would be $$\mathbb P(\mathcal X_1=a, \mathcal X_2=d, \mathcal X_3=b, \mathcal X_4=c).$$ The following code uses the probability chain rule to estimate the likelihood.

```python
text = 'a d b c'
lst = [w2id[x] for x in text.split()]
p = M_r[lst[0]]
for a, b in zip(lst, lst[1:]):
    p *= p_l[a, b]
```

The previous example is complemented with a sequence that, by definition, it is known that has a lower probability, the sequence differs only in the last word, and its probability is $$\mathbb P(\text{"a d b d"}) = 0.006907.$$

The procedure described presents the process of modeling a language from the beginning; it starts by assuming that the language is generated from a particular algorithm, then the algorithm is used to estimate a bivariate distribution, which is used to produce a sequence. The sequence is an analogy of a text written in natural language, then the sequence is used to estimate a bivariate distribution, and we can compare both distributions to illustrate that even in a simple process, it is unfeasible to obtain two matrices with the same values. 

# Overcoming Assumptions

However, some components of the previous formulation are unrealistic for modeling a language. The first one is the addition of the possible sequences of a particular length sum to one, e.g., $$\sum_{x,y,z} \mathbb P(\mathcal X_{\ell-2}=x, \mathcal X_{\ell-1}=y, \mathcal X_\ell=z) = 1$$. The implication is that there is a probability distribution for every length, which is not a desirable feature for a language model because the length of a sentence is variable. 

The second one is that the first word cannot be estimated using the marginal $$\mathbb P(\mathcal X_r)$$; this distribution does not consider that some words are more frequently used to start a sentence. This effect can be incorporated in the model with a starting symbol, e.g., the sequence *a b d* would be *$$\epsilon$$ a b d*. 

The sum of the probabilities of all possible sentences with three words and a starting symbol can be seen in the following equation. 

$$
\begin{eqnarray}
&&\sum_{x, y, z} \mathbb P(\mathcal X_1=\epsilon, \mathcal X_2=x, \mathcal X_3=y, \mathcal X_4=z) \approx \\
&&\sum_{x, y, z} \mathbb P(\mathcal X_4=z \mid \mathcal X_3=y) \mathbb P(\mathcal X_3=y \mid \mathcal X_2=x) \mathbb P(\mathcal X_2=x \mid \mathcal X_1=\epsilon) \mathbb P(\mathcal X_1=\epsilon) =\\
&&\mathbb P(\mathcal X_1=\epsilon) \sum_{x, y, z} \mathbb P(\mathcal X_4=z \mid \mathcal X_3=y) \mathbb P(\mathcal X_3=y \mid \mathcal X_2=x) \mathbb P(\mathcal X_2=x \mid \mathcal X_1=\epsilon) =\\
&&\mathbb P(\mathcal X_1=\epsilon) \sum_{x, y}  \mathbb P(\mathcal X_3=y \mid \mathcal X_2=x) \mathbb P(\mathcal X_2=x \mid \mathcal X_1=\epsilon)\sum_z \mathbb P(\mathcal X_4=z \mid \mathcal X_3=y) =\\
&&\mathbb P(\mathcal X_1=\epsilon) \sum_{x}  \mathbb P(\mathcal X_2=x \mid \mathcal X_1=\epsilon) \sum_y  \mathbb P(\mathcal X_3=y \mid \mathcal X_2=x)=\\
&&\mathbb P(\mathcal X_1=\epsilon) \sum_{x}  \mathbb P(\mathcal X_2=x \mid \mathcal X_1=\epsilon) = \mathbb P(\mathcal X_1=\epsilon) = 1
\end{eqnarray}
$$

It can be observed that the inclusion of a starting symbol does not solve the problem that there is a probability distribution for every length.

The third problem detected is that the length of the sentence is a parameter controlled by the person generating the sequence; however, the length of the sentence depends on the content of the sentence, so it is also a random variable. A feasible approach to include this behavior in the model is adding an ending symbol. 

The accumulated probability of all the possibilities for a three-word sentence with a starting symbol and ending symbols (i.e., $$\epsilon_s$$ and $$\epsilon_e$$, respectively), can be expressed as follows:


$$
\begin{eqnarray}
&&\sum_{x, y, z} \mathbb P(\mathcal X_1=\epsilon_s, \mathcal X_2=x, \mathcal X_3=y, \mathcal X_4=z, \mathcal X_5=\epsilon_e) \approx \\
&&\mathbb P(\mathcal X_1=\epsilon_s) \sum_{x, y, z} \mathbb P(\mathcal X_5=\epsilon_e \mid \mathcal X_4=z) P(\mathcal X_4=z \mid \mathcal X_3=y) \mathbb P(\mathcal X_3=y \mid \mathcal X_2=x) \mathbb P(\mathcal X_2=x \mid \mathcal X_1=\epsilon)  =\\
\end{eqnarray}
$$




<!--
A language model is a model that assigns probabilities to words (tokens). That is, the aim is to estimate the probability of the next token using the history. The simplest case is $$P(w_m \mid  w_{m-1})$$ where the probability of token $$w_m$$ is only influence with the previous token $$w_{m-1}$$. However, this case can be easily extended to compute $$P(w_m \mid  w_1, w_2, \ldots, w_{m-1})$$.

The essential elements of an N-Gram Language Model (LM) are the number of times the tokens and n-grams appear in a corpus; with this information, it is possible to compute the probability of a particular sentence and generate sentences using the LM. One way to start creating the model is by storing variables containing these counts accessible to all the methods in a class. Let us create a class NgramLM, that stored the counting information as well as the instance used to tokenize a text, .i.e., nlp.

```python
import re
from typing import Tuple, List
from collections import Counter

class NgramLM(object):
    def __init__(self, n: int=2, nlp: object=None) -> None:
        from spacy.lang.en import English
        self._tokens = Counter()
        self._n_grams = Counter()
        self.n = n
        if nlp is None:
            self.nlp = English()
```

# Training a Language Model

The procedure to obtain the frequency of the tokens and n-grams is to process a corpus and count the number of appearances, being the first step to convert the text into tokens and n-grams. The step can be performed with the following code:

```python
def tokenize(self, txt: str) -> List[str]:
    _ = [x.norm_.strip() for x in self.nlp(txt)]
    _ = [x for x in _ if len(x)]
    if len(_) == 0:
        return _
    _.insert(0, "<p>")
    _.append("</p>")
    return _
```

The previous code receives a text, and returns a list containing the tokens of the text; it relies on the use of a tokenizer (nlp). The tokens \<p\> and \</p\> indicate the start and end of the text. It is essential to note that it is coded for bigrams and does not consider $n>2$. 

Considering that the corpus can be a list of texts -where the texts can be sentences or paragraphs- and that we have already defined a method to tokenize a text, it is time to count the appearances of the tokens and n-grams using as input a list of texts.

```python
def process_paragraphs(self, para: List[str]):
    for p in para:
        _ = self.tokenize(p)
        self._tokens.update(_)
        _ = self.n_grams(_, n=self.n)
        self._n_grams.update(_)
```

It can be observed that the list of tokens is transformed to n-grams using the following function.

```python
def n_grams(tokens: list, n: int):
    ww = [tokens[i:] for i in range(n)]
    _ = ["~".join(x) for x in zip(*ww)]
    return _
```

The last step, which is necessary given that we are working using the books from Project Gutenberg is to process a text file and split it into paragraphs.

```python
def process_file(self, fname: str):
    txt = open(fname).read()
    para = [x for x in re.finditer(r"\n\n", txt)]
    index = [0] + [x.end(0) for x in para]
    para = [txt[i:j] for i, j in zip(index, index[1:])]
    self.process_paragraphs([x for x in para if len(x) > 2])
```

# Using the LM 

Once the frequency is computed, it is time to measure $P(w_m \mid w_{m-1})$ the probability of a particular n-gram. It is important to note that the following code only handles bigrams.

```python
def prob(self, n_gram: str) -> float:
    c_bi = self._n_grams[n_gram]
    a, _ = n_gram.split("~")
    c_token = self._tokens[a]
    return c_bi / c_token
```

The previous code can be used to compute the probability of a sentence, i.e., $P(w_1,w_2, \ldots, w_m) = \prod_{k=1}^m P(w_{m-n+1}, \ldots, w_{k-2}, w_{k-1})$ as follows.

```python
def sentence_prob(self, txt: str) -> float:
    tokens = self.tokenize(txt)
    ngrams = self.n_grams(tokens, n=self.n)
    p = 1
    for x in ngrams:
        p = p * self.prob(x)
    return p
```

The code review during class is missing two key features. One is the possibility to compute the $\log P(w_1, w_2, \ldots, w_m)$, and the second is that it can only handle bigrams. The homework deals with these two problems as well as testing the algorithm to generate sentences.

The first part is to create the method to compute $\log P(w_1, w_2, \ldots, w_m)$. The method should receive a sentence as a string, and compute the log probability. The method must verify that the sentence tokenize has the minimum number of tokens to create at least one n-gram. For example, for bigrams the tokenize sentence must be greater or equal to two, for 3-gram it is greater or equal to three, and so on. The method being overwritten is `log_sentence_prob`. The following code shows the method computing the probability and the one that needs to be implemented.

```python
def sentence_prob(self, txt: str, markers: bool=False) -> float:
    """Probability of a sentence P(w_1, w_2, ..., w_n)
    
    :param txt: text
    :param markers: include starting and ending markers
    :type markers: bool        

    >>> ngram = NgramLM()
    >>> ngram.process_paragraphs(["xxx xyx xxy", "xyx aaa xxx"])
    >>> ngram.sentence_prob("xxx xyx aaa")
    0.25
    """
    tokens = self.tokenize(txt, markers=markers)
    ngrams = self.n_grams(tokens)
    p = 1
    for x in ngrams:
        _ = self.prob(x)
        p = p * _
    return p

def log_sentence_prob(self, txt: str, markers: bool=True) -> float:
    pass
```

Note that there is an extra parameter, markers, this is to include the start of the sentence or just to consider that it is a text in the middle of a sentence or paragraph. 

The second part is to extend the algorithm to deal with n-grams greater than 2. The starting code already receives as an argument the size of the grams; however, it can only deal with bigrams. 

It is essential to note that the tokens are needed to generate the sentences, and that $$P(w_n \mid w_{n-k}, \ldots, w_{n-2}, w_{n-1})$$ is based on $$C(w_{n-k}, \ldots, w_{n-2}, w_{n-1})$$ and $$C(w_{n-k}, \ldots, w_{n-2}, w_{n-1}, w_n)$$.

-->