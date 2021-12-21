---
layout: default
title: Text Processing
nav_order: 2
---

# Text Processing
{: .fs-10 .no_toc }

The **objective** is to 

## Contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---

# Introduction

The journey of natural language processing starts with the simple procedure of counting words; with it, we will be able to model some characteristics of the languages, identify patterns in a text, and perform exploratory data analysis on a collection of Tweets. 

At this point, let us define a word as a sequence of characters bounded by a space - this is a shallow definition; however, it is suitable for most words written in Latin languages and English.

## Frequency of words

The frequency of words in a document can be computed using a dictionary. A dictionary is a data structure that associates a keyword with a value. The following code uses a dictionary (`word`) to count the word frequencies of texts stored in a JSON format, each one per line. It uses the function `tweet_iterator` that iterates over the file, scanning one line at a time and converting the JSON into a dictionary where the keyword text contains the text.

```python
from microtc.utils import tweet_iterator
from EvoMSA.tests.test_base import TWEETS

words = dict()
for tw in tweet_iterator(TWEETS):
    text = tw['text']
    for w in text.split():
        key = w.strip()
        try:
            words[key] += 1
        except KeyError:
            words[key] = 1
```

Method `split` returns a list of strings where the split occurs on the space character, which follows the definition of a word. Please note the use of an exception to handle the case where the keyword is not in the dictionary. As an example, the word _si_ (yes in Spanish) appears 29 times in the corpus.

```python
words['si']
29
```

The counting pattern is frequent, so it is implemented in the Counter class under the package collections; the following code implements the method described using it; the key element is the method `update,` and to make the code shorter, [list comprehension](https://docs.python.org/3/tutorial/datastructures.html#list-comprehensions) is used. 

```python
from microtc.utils import tweet_iterator
from EvoMSA.tests.test_base import TWEETS
from collections import Counter

words = Counter()
for tw in tweet_iterator(TWEETS):
    text = tw['text']
    words.update([x.strip() for x in text.split()])
```

Counter class has another helpful method to analyze the frequencies; in particular, the `most_common` method returns the most frequent keywords. For example, the following code gets the five most common keywords. 

```python
words.most_common(5)
[('de', 299), ('que', 282), ('a', 205), ('la', 182), ('el', 147)]
```

These are the words, _of_, _what_, _a_, _the_, and _the_, respectively.

# Herdan’s Law / Heaps' Law

# Regular Expressions

The problem of named-entity recognition is an NLP task that deals with identifying the entities such as person names, organizations, locations, among others. For example, in "Juan bought a MacBook," the idea is to identify that Juan  is a person, and MacBook is a laptop. This problem has been tackled using different techniques, and the approach followed consists of two steps. The first one is identifying the entity, and the second one corresponds to identifying the types, that is, to know whether the entity is a person name, organization, location, among other types.  

A simple heuristic used to identify entities is to mark as an entity each word starting with a capital letter or an abbreviation. The objective is to design a regular expression that captures these patterns and counts the number of entities found in a given text. The following code shows the structure of the function that needs to be implemented and some examples.

The text normalization techniques used depend on the application; for example, when analyzing social media (e.g., Twitter), there are occasions that it is desired to remove all the mentions of the users in order to keep their privacy; nonetheless, it could be helpful to indicate that a username was present. The following exercise consists in creating a function that receives a text and replaces all the appearances of a username with the tag @user.

An essential component when developing an algorithm is to measure its performance. The performance has different forms depending on the algorithm; for example, the performance could be the algorithms' complexity in a sorting algorithm. On the other hand, the performance on machine learning corresponds to measure the similarity between a correct answer and the produce by the algorithm, or equivalent can be an error function. 

This exercise asks to create a function that computes the performance of a sentence tokenizer. The performance measure used is accuracy which is defined as the percentage of correct answers. Let $$y$$ and $$\hat y$$ be the list of correct sentences and the sentences obtained with the tokenizer. Let $$\delta (x)$$ be a function that returns 1 when $$x$$ is true and 0 otherwise. Using these elements the accuracy can be computed as: 

$$\frac{1}{\mid y \mid} \sum_{x \in y} \delta(x \in \hat y),$$

where the operation $$x \in \hat y$$ deletes the sentence $$x$$ from $$\hat y$$ once it has been tested; this is not normal behavior but it is necessary to compute the accuracy in this situation.


# Collocations


