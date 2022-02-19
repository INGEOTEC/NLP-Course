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

## Users
## URL
## Numbers

# Spelling

## Punctuation
## Diactric
## Case sensitive
## Symbol reduction

# Stopwords

# Stemmming

# Lemmatization

# Tokenization

## q-grams

## n-grams


