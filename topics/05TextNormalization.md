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
re.sub(r"(\d+\.\d+)|(\.\d+)|(\d+\.)|(\d+)", "_num", text)
```

# Spelling

The next block of text normalization modifies the writing of the text, removing components that, for particular applications, can be ignored to reduce the vocabulary size, which impacts the complexity of the algorithm and could be reflected in an improvement in the performance.  

## Case sensitive

The first of these transformations is the conversion to lower case; transforming all the words to the lower case has the consequence that the vocabulary is reduced, e.g., the word Mexico and mexico would be considered the same token. This operation can be implemented with function `lower` as follows.

```python
text = "Mexico"
text.lower()
```

## Punctuation

The punctuation symbols are essential to natural language understanding and generation; however, for other applications, such as sentiment analysis or text categorization, its contribution is opaque by the increase in the vocabulary size. Consequently, its removal influences the vocabulary size, which sometimes has a positive result on the performance.

These symbols can be removed by traversing the string and skipping the punctuations.

```python
text = "Hi! good morning,"
output = ""
for x in text:
    if x in SKIP_SYMBOLS:
        continue
    output += x
```

## Diacritic

Different languages use diacritic symbols, e.g., México; as expected, this has the consequence of increasing the vocabulary. On the other hand, in informal writing, the misuse of diacritic symbols is common; one particular way to handle this problem is to remove the diacritic symbols and treat them as the same word, e.g., México would be replaced by Mexico. 

```python
text = 'México'
output = ""
for x in unicodedata.normalize('NFD', text):
    o = ord(x)
    if 0x300 <= o and o <= 0x036F:
        continue
    output += x
```

# Semantic Normalizations

The next set of normalization techniques aims to reduce the vocabulary size using the meaning of the words to modify them or remove them from the text.

## Stop words

The stop words are the most frequent words used in the language. These words are essential to communicate but are not so much on tasks where the aim is to discriminate texts according to their meaning. 

The stop words can be stored in a dictionary, and then the process of removing them consists of traversing all the tokens from a text and then removing those in the dictionary. The process is exemplified with the following code.

```python
lang = LangDependency('english')

text = 'Good morning! Today, we have a warm weather.'
output = []
for word in text.split():
    if word.lower() in lang.stopwords[len(word)]:
        continue
    output.append(word)
output = " ".join(output) 
```

## Stemmming and Lemmatization

The idea of stemming and lemmatization, seen as a normalization process, is to group different words based on their root; for example, the process would associate words like *playing*, *player*, *plays* with the token *play*.

Stemming treats the problem with fewer constraints than lemmatization, having as a consequence that the common word found cannot be the common root of the words; additionally, the algorithms do not consider the role of the word being processed in the sentence. On the other hand, a lemmatization algorithm obtains the root of the word considering the part of the speech of the processed word.

```python
stemmer = PorterStemmer()

text = 'I like playing football'
output = []
for word in text.split():
    w = stemmer.stem(word)
    output.append(w)
output = " ".join(output) 
output
```

# Tokenization

## q-grams

## n-grams


