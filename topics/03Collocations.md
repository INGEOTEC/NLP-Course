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

# Collocations

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

The idea is to use the information of the co-occurrence matrix to find the pairs of words that can be considered collocations. As can be observed, the co-occurrence matrix can be transformed into a bivariate distribution and use a statistical approach to retrieve some prominent pairs. Before going into the details of these algorithms, it is necessary to describe how the words in the co-occurrence matrix are considered random variables. 

Each element in the matrix can be uniquely identified by the pair words, e.g., the frequency of pair (_in_, _of_) is $$122502$$. However, it is also possible to identify the same element using an index. For example, if the first word (_the_) is assigned the index $$0$$, the index $$4$$ corresponds to word _in_ and $$2$$ to _of_. Consequently, the element (_in_, _of_) can uniquely identify with the pair (4, 2). One can create a mapping between words and natural numbers such that each different word has a unique identifier. The mapping allows working with natural numbers instead of words which facilitates the analysis and returns to the words (using the inverse mapping) when the result is obtained. 

The mapping can be implemented using a dictionary, as seen from the following code where the variable of interest is `index.` 

```python
index = dict()
for bigram, cnt in bigrams.items():
    a, b = bigram.split('~')
    for x in [a, b]:
        if x not in index:
            index[x] = len(index)
len(index)
```


