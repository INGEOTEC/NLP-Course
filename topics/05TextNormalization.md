---
layout: default
title: Text Normalization
nav_order: 5
---

# Text Normalization
{: .fs-10 .no_toc }

## Contents
{: .no_toc .text-delta }

1. TOC
{:toc}

## Libraries used
{: .no_toc .text-delta }
```python
import numpy as np
from wordcloud import WordCloud as WC
from matplotlib import pylab as plt
from microtc.textmodel import TextModel
```

---

# Introduction

In all the topics covered, the assumption is that the text is well-formatted and spaces nicely surround the words (tokens). However, this is not the general case, and the spelling errors and the procedure used to define the tokens strongly impact the algorithm's performance. Consequently, this part of the course is devoted to presenting standard techniques used to normalize the text and to transform the text into tokens. 

The text normalization described are mainly the ones used in the following research words:

1. [An automated text categorization framework based on hyperparameter optimization](https://www.sciencedirect.com/science/article/pii/S0950705118301217)
2. [A simple approach to multilingual polarity classification in Twitter](https://www.sciencedirect.com/science/article/abs/pii/S0167865517301721)
3. [A case study of Spanish text transformations for twitter sentiment analysis](https://www.sciencedirect.com/science/article/abs/pii/S0957417417302312)

# Entity

The journey of text normalization starts with handling different entities within a text; the entities could be the mentioned of a user in a tweet, the numbers, or the URL, to mention a few. The actions performed to the entities found are to delete them or replace them for a particular token. 

## Users

The first process is to deal with username following the format of Twitter. In a tweet, the mention of a user is identified with a string starting with the character @. The two actions could be to delete all the users' mentions or change them for a common label.

The procedure uses regular expressions to find the entities; for example, the following code can remove the users' mentions.

```python
text = 'Hi @xx, @mm is talking about you.'
re.sub(r"@\S+", "", text)
'Hi   is talking about you.'
```

On the other hand, to replace the username with a shared label can be implemented with the following code, where the label is `_usr`

```python
text = 'Hi @xx, @mm is talking about you.'
re.sub(r"@\S+", "_usr", text)
'Hi _usr _usr is talking about you.'
```

## URL

The previous code can be adapted to handle URL; one only needs to define the regular expression to use; see the following code that removes all the appearances of the URL. 

```python
text = "go http://google.com, and find out"
re.sub(r"https?://\S+", "", text)
```

## Numbers

The previous code can be modified to deal with numbers and replace the number found with a shared label such as `_num`.

```python
text = "we have won 10 M"
re.sub(r"\d+\.?\d+", "_num", text)
```

# Spelling

## Case sensitive
## Punctuation
## Diactric
## Symbol reduction

# Stopwords

# Stemmming

# Lemmatization

# Tokenization

## q-grams

## n-grams


