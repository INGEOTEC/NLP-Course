---
layout: default
title: N-Gram Language Model
nav_order: 5
---

# N-Gram Language Model
{: .fs-10 .no_toc }

The **objective** is to 

## Contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---

# Introduction

A language model is a model that assigns probabilities to the tokens. That is, the aim is to estimate the probability of the next token using the history. The simplest case is $$P(w_m \mid  w_{m-1})$$ where the probability of token $$w_m$$ is only influence with the previous token $$w_{m-1}$$. However, this case can be easily extended to compute $$P(w_m \mid  w_1, w_2, \ldots, w_{m-1})$$.

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