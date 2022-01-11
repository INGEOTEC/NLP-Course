---
layout: default
title: Text Processing
nav_order: 2
---

# Text Processing
{: .fs-10 .no_toc }

## Contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---

# Introduction

The journey of natural language processing starts with the simple procedure of counting words; with it, we will be able to model some characteristics of the languages, identify patterns in a text, and perform exploratory data analysis on a collection of Tweets. 

At this point, let us define a word as a sequence of characters bounded by a space - this is a shallow definition; however, it is suitable for most words written in Latin languages and English.

## Frequency of words

The frequency of words in a document can be computed using a dictionary. A dictionary is a data structure that associates a keyword with a value. The following code uses a dictionary (variable `word`) to count the word frequencies of texts stored in a JSON format, each one per line. It uses the function `tweet_iterator` that iterates over the file, scanning one line at a time and converting the JSON into a dictionary where the keyword text contains the text.

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

The counting pattern is frequent, so it is implemented in the [Counter](https://docs.python.org/3/library/collections.html#collections.Counter) class under the package [collections](https://docs.python.org/3/library/collections.html); the following code implements the method described using `Counter`; the key element is the method `update,` and to make the code shorter, [list comprehension](https://docs.python.org/3/tutorial/datastructures.html#list-comprehensions) is used. 

```python
from microtc.utils import tweet_iterator
from EvoMSA.tests.test_base import TWEETS
from collections import Counter

words = Counter()
for tw in tweet_iterator(TWEETS):
    text = tw['text']
    words.update([x.strip() for x in text.split()])
```

`Counter` has another helpful method to analyze the frequencies; in particular, the `most_common` method returns the most frequent keywords. For example, the following code gets the five most common keywords. 

```python
words.most_common(5)
[('de', 299), ('que', 282), ('a', 205), ('la', 182), ('el', 147)]
```

These are the words, _of_, _what_, _a_, _the_, and _the_, respectively.

# Zipf's Law

The word frequency allows defining some empirical characteristics of the language. These characteristics are essential to understand the challenges of developing an algorithm capable of understanding language. 

Let us start with Zipf's law. The law relates the frequency of a word with its rank. In an order set, the rank corresponds to the element's position, e.g., the first element has the rank of one, the second has the second rank, and so on.  Explicitly, the relationship is defined as $$f \cdot r = c $$, where $$f$$ is the frequency, $$r$$ is the rank, and $$c$$ is a constant. For example, the frequency is $$f=\frac{c}{r}$$; as can be seen when the rank equals the constant, then the frequency is one, meaning that the rest of the words are infrequent and that there are only frequent few words. 

The following figure depicts the described characteristic. It shows a scatter plot between rank and frequency. 

![Zipf's Law](/NLP-Course/assets/images/zipf_law.png)

The frequency and the rank can be computed with the following two lines using the dataset of the previous example. 

```python
freq = [f for _, f  in words.most_common()]
rank = range(1, len(freq) + 1)
```

Once the frequency and the rank are computed, the figure is created using the following code. 

```python
from matplotlib import pylab as plt
plt.plot(rank, freq, '.')
plt.grid()
plt.xlabel('Rank')
plt.ylabel('Frequency')
plt.tight_layout()
```

The previous figure does not depict the relation $$f=\frac{c}{r}$$, in order to show it let us draw the figure scatter plot between the rank inverse and the frequency; this can be obtained by changing the following lines into the previous procedure -- [numpy](https://numpy.org) is used in the following code to create an array of elements.

```python
import numpy as np
freq = [f for _, f  in words.most_common()]
rank = 1 / np.arange(1, len(freq) + 1)
```

As observed from the following figure, Zipf’s Law is a rough estimate of the relationship between rank and frequency. Nonetheless, it provides all the elements to understand the importance of this relationship. 

![Log Zipf's Law](/NLP-Course/assets/images/zipf_law2.png) 

## Ordinary Least Squares

The missing step is to estimate the value of $$c$$. Constant $$c$$ can be calculated using ordinary least squares (OLS). The idea is to create a system of equations where the unknown is $$c$$, and the dependent variable is $$f$$. These can represent in matrix notation as $$A \cdot \mathbf c = \mathbf f$$, where $$A \in \mathbb R^{N \times 1}$$ is composed by $$N$$ inverse rank measurements, $$c \in \mathbb R^1$$ is the parameter to be identified, and $$\mathbf f \in \mathbb R^N$$ is a column vector containing the frequency measurements.

The coefficients can be computed using the function `np.linalg.lstsq` described in the following code.

```python
X = np.atleast_2d(rank).T
c = np.linalg.lstsq(X, freq, rcond=None)[0]
c
array([461.40751913])
```

Once $$c$$ has been identified, it can be used to predict the model; the following figure presents the measurements and the predicted points using the identified coefficient.

![Zipf's Law - model](/NLP-Course/assets/images/zipf_law3.png)

The previous figure was created with the following code; the variable `hy` contains the predicted frequency.

```python
hy = np.dot(X, c)
plt.plot(rank, freq, '.')
plt.plot(rank, hy)
plt.legend(['Measured', 'Predicted'])
plt.grid()
plt.xlabel('Inverse Rank')
plt.ylabel('Frequency')
plt.tight_layout()
```

# Herdan’s Law / Heaps' Law

A language used evolves new words are incorporated in the language, and the relationship between the vocabulary size and the number of words (tokens) is expressed in the Heaps' Law. Let $$\mid V \mid$$ represents the vocabulary size, and $$n$$ the number of words; then the relationship between these elements is $$\mid v \mid = k n^\beta$$ where $$k$$ and $$\beta$$ are the parameters that need to be identified. 

The following figure depicts the relation between $$n$$ and $$\mid v \mid$$ using the dataset of the previous examples.

![Heaps' Law](/NLP-Course/assets/images/heaps_law.png) 

Taking the procedure used in Zipf's Law as a starting point, it is only needed to collect, for each line, the number of words and the vocabulary.

```python
words = Counter()
tokens_voc= list()
for tw in tweet_iterator(TWEETS):
    text = tw['text']
    words.update([x.strip() for x in text.split()])
    tokens_voc.append([sum(list(words.values())),
                       len(words)])

n = [x[0] for x in tokens_voc]
v = [x[1] for x in tokens_voc]
```

Once the points (number of words and vocabulary size) are measured, it is time to estimate the parameters $$k$$ and $$\beta$$. As can be observed, it is not possible to express the Heaps' Law as a system of equations such as $$A \cdot [k, \beta]^\intercal = \mid \mathbf v \mid$$; consequently, these parameters cannot be estimated using OLS.

## Optimization

The values of these parameters can be estimated by posing them as an optimization problem. An optimization problem minimizes (maximizes) an objective function. The goal is to find the inputs corresponding to the minimum (maximum) of the function, i.e., 

$$\min_{\mathbf w \in \Omega} f(\mathbf w),$$

where $$\mathbf w$$ are the inputs of the function $$f$$, and $$\Omega$$ is the search space, namely the set containing all feasible values of the inputs, e.g., $$\Omega = \{\mathbf w \mid \mathbf w \in \mathbb R^d, \forall_i \mathbf w_i \geq 0\}$$ that represents all the vector of dimension $$d$$ whose components are equal or greater than zero. 

There are many optimization algorithms; some are designed to tackle all possible optimization problems, and others are tailored for a particular class of problems. For example, OLS is an optimization algorithm where the objective function $$f$$ is defined as:

$$f(\mathbf w) = \sum_{i=1}^N (y_i - \mathbf a_i \cdot \mathbf w)^2,$$

where $$\mathbf w$$ is a vector containing the parameters, and $$\mathbf a_i$$ is the $$i$$-th measurement of the independent variables, and $$y_i$$ is the corresponding dependent variable. In the Zipf's Law example, $$\mathbf w$$ corresponds to the parameter $$c$$, $$\mathbf a_i$$ is the $$i$$-the measure of the inverse rank, and $$y_i$$ is the corresponding frequency. 

Let us analyze in more detail the previous objective function. The sum goes for all the measurements; there are $$N$$ pairs of observations composed by the response (dependent variable) and the independent variables. For each observation $$i$$, the square error is computed between the dependent variable ($$y_i$$) and the dot product of the independent variable ($$\mathbf a_i$$) and parameters ($$\mathbf x$$). 

This notation is specific for OLS; however, it is helpful to define the general case. The first step is to define the set $$\mathcal X$$ composed by the $$N$$ observations, i.e., $$\mathcal X=\{(y_i, \mathbf x_i) \mid 1 \leq i \leq N\}$$ where $$y_i$$ is the response variable and $$\mathbf x$$ contains the independent variables. The second step is to  use an unspecified loss function as $$L(y, \hat y) = (y - \hat y)^2$$. The last step is to abstract the model, that is replace $$\mathbf a_i \cdot \mathbf w$$ with a function $$g(\mathbf x) = \mathbf a_i \cdot \mathbf w$$.  Combining these components the general version of the OLS optimization problem can be expressed as:

$$\min_{g \in \Omega} \sum_{(y, \mathbf x) \in \mathcal X} L(y, g(\mathbf x)),$$
 
where $$g$$ are all the functions defined in the search space $$\Omega$$, $$L$$ is a loss function, and $$\mathcal X$$ contains $$N$$ pairs of observations. This optimization problem is known as **supervised learning** in machine learning.

Returning to Heaps' Law problem where the goal is to identify the coefficients $$k$$ and $$\beta$$ of the model $$\mid v \mid = kn^\beta$$. That is, function $$g$$ can be defined as $$g_{k,\alpha}(n) = kn^\beta$$ and using the square error as $$L$$ the optimization problem is:

$$\min_{(k, \beta) \in \mathbb R^2} \sum_{(\mid v \mid, n) \in \mathcal X} (\mid v \mid -  kn^\beta)^2.$$

As mentioned previously, there are many optimization algorithms; some can be found on the function [scipy.optimize.minimize](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html). The function admits as arguments the objective function and the initial values of the parameters. 

The following code solves the optimization problem using the `minimize` function. 

```python
from scipy.optimize import minimize
n = np.array(n)
v = np.array(v)
def f(w, y, x):
    k, beta = w
    return ((y - k * x**beta)**2).sum()

res = minimize(f, np.array([1, 0.5]), (v, n))
k, beta = res.x
k, beta
(2.351980550243793, 0.8231559504194587)
```

Once $$k$$ and $$\beta$$ have been identified, it is possible to use them in the Heaps' Law and produce a figure containing the measurements and predicted values.

![Heaps' Law - model](/NLP-Course/assets/images/heaps_law2.png)


# Activities

Zipf's Law and Heaps' Law model two language characteristics; these characteristics are summarized in the values of the parameters $$c$$ and $$k$$ and $$\beta$$ respectively.  

```python
from text_models import Vocabulary
date = dict(year=2022, month=1, day=10)
voc = Vocabulary(date, lang='Es', country='MX')
words = {k: v for k, v in voc.voc.items() if not k.count('~')}
```

![Word Cloud (MX)](/NLP-Course/assets/images/wordcloud_mx.png)


```python
from wordcloud import WordCloud as WC
wc = WC().generate_from_frequencies(words)
plt.imshow(wc)
plt.axis('off')
plt.tight_layout()
```



<!---

# Regular Expressions

The problem of named-entity recognition is an NLP task that deals with identifying the entities such as person names, organizations, locations, among others. For example, in "Juan bought a MacBook," the idea is to identify that Juan  is a person, and MacBook is a laptop. This problem has been tackled using different techniques, and the approach followed consists of two steps. The first one is identifying the entity, and the second one corresponds to identifying the types, that is, to know whether the entity is a person name, organization, location, among other types.  

A simple heuristic used to identify entities is to mark as an entity each word starting with a capital letter or an abbreviation. The objective is to design a regular expression that captures these patterns and counts the number of entities found in a given text. The following code shows the structure of the function that needs to be implemented and some examples.

The text normalization techniques used depend on the application; for example, when analyzing social media (e.g., Twitter), there are occasions that it is desired to remove all the mentions of the users in order to keep their privacy; nonetheless, it could be helpful to 2indicate that a username was present. The following exercise consists in creating a function that receives a text and replaces all the appearances of a username with the tag @user.

An essential component when developing an algorithm is to measure its performance. The performance has different forms depending on the algorithm; for example, the performance could be the algorithms' complexity in a sorting algorithm. On the other hand, the performance on machine learning corresponds to measure the similarity between a correct answer and the produce by the algorithm, or equivalent can be an error function. 

This exercise asks to create a function that computes the performance of a sentence tokenizer. The performance measure used is accuracy which is defined as the percentage of correct answers. Let $$y$$ and $$\hat y$$ be the list of correct sentences and the sentences obtained with the tokenizer. Let $$\delta (x)$$ be a function that returns 1 when $$x$$ is true and 0 otherwise. Using these elements the accuracy can be computed as: 

$$\frac{1}{\mid y \mid} \sum_{x \in y} \delta(x \in \hat y),$$

where the operation $$x \in \hat y$$ deletes the sentence $$x$$ from $$\hat y$$ once it has been tested; this is not normal behavior but it is necessary to compute the accuracy in this situation.
-->
