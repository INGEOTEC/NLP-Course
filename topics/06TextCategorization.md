---
layout: default
title: Text Categorization
nav_order: 6
---

# Text Categorization
{: .fs-10 .no_toc }

## Contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---

# Introduction

Text Categorization is an NLP task that deals with creating algorithms capable of identifying the category of a text from a set of predefined categories. For example, sentiment analysis belongs to this task, and the aim is to detect the polarity (e.g., positive, neutral, or negative) of a text. Furthermore, different NLP tasks that initially seem unrelated to this problem can be formulated as a classification one such as question answering and sentence entailment, to mention a few. 

Text Categorization can be tackled from different perspectives; the one followed here is to treat it as a supervised learning problem. As in any supervised learning problem, the starting point is a set of pairs, where the first element of the pair is the input and the second one corresponds to the output. Let $$\mathcal D = \{(\text{text}_i, y_i) \mid i=1,\ldots, N\}$$ where $$y \in \{c_1, \ldots c_K\}$$ and $$\text{text}_i$$ is a text. 

Supervised learning problems can be seen as finding a mapping function from inputs to outputs. The tool could be an [optimization](/NLP-Course/topics/02Vocabulary/#sec:optimization) algorithm capable of finding the function that minimizes a particular loss function, e.g., $$L$$. 

$$\min_{g \in \Omega} \sum_{(\mathbf x, y) \in \mathcal D} L(y, g(\mathbf x)),$$

where $$\Omega$$ is the search space of the feasible mapping functions.

Additionally, if one is also interested in measuring the uncertainty, the path relies on the probability. In this latter scenario, one approach is to assume the form of the conditional probability, i.e., $$\mathbb P(\mathcal Y=k \mid \mathcal X=x)=f_k(x)$$ where $$f_k$$ is the $$k$$-th value of $$f: \mathcal X \rightarrow [0, 1]^K$$ which encodes a probability mass function. For the case of a binary classification problem the function is $$f: \mathcal X \rightarrow [0, 1]$$. As can be seen, in this scenario, an adequate distribution is [Bernoulli](/NLP-Course/topics/03Collocations/#sec:bernoulli), where function $$f$$ takes the place of the parameter of the distribution, that is, $$\mathcal Y \sim \textsf{Bernoulli}(f(\mathcal X))$$; for more labels, the Categorical distribution can be used. The complement path is to rely on Bayes' theorem. 

# Bayes' theorem
{: #sec:bayes-theorem }

The bivariate distribution $$\mathbb P(\mathcal X, \mathcal Y)$$ can be expressed using the [conditional probability](/topics/04NGramLM/#sec:conditional-probability) as:

$$\begin{eqnarray}
\mathbb P(\mathcal X, \mathcal Y) &=& \mathbb P(\mathcal X \mid \mathcal Y) \mathbb P(\mathcal Y)\\
\mathbb P(\mathcal X, \mathcal Y) &=& \mathbb P(\mathcal Y \mid \mathcal X) \mathbb P(\mathcal X).
\end{eqnarray}$$

These elements can be combined to obtain Bayes' theorem following the next steps: 

$$\begin{eqnarray}
\mathbb P(\mathcal Y \mid \mathcal X) \mathbb P(\mathcal X) &=& \mathbb P(\mathcal X \mid \mathcal Y) \mathbb P(\mathcal Y)\\
\mathbb P(\mathcal Y \mid \mathcal X)  &=& \frac{\mathbb P(\mathcal X \mid \mathcal Y) \mathbb P(\mathcal Y)}{\mathbb P(\mathcal X)},
\end{eqnarray}$$

where $$\mathbb P(\mathcal Y \mid \mathcal X)$$ is the **posterior probability**, $$\mathbb P(\mathcal X \mid \mathcal Y)$$ corresponds to the **likelihood**, $$\mathbb P(\mathcal Y)$$ is the **prior**, and $$\mathbb P(\mathcal X)$$ is the **evidence**. The evidence can be expressed as: $$\mathbb P(\mathcal X) = \sum_y \mathbb P(\mathcal X \mid \mathcal y) \mathbb P(\mathcal y)$$ which acts as a normalization constant.


We have seen how to use Naive Bayes to solve a classification problem; that is, we start with a set $$\{(\mathbf x_i, y_i) \mid i=1, \ldots, N\}$$ where $$\mathbf x_i \in \mathbb R^d$$, this set is then used to obtain the parameters of a Naive Bayes classifier, and finally, we can use the trained model to test points that have not been used in the training process.

It is important to note that in order to use a classifier such as Naive Bayes, the inputs must have to be represented as vectors, i.e., $\mathbf x \in R^d$. However, in text categorization the inputs are texts.

## Classification

$$\{(\mathbf x_i, y_i) \mid i=1, \ldots, N\}$$ 
where $$\mathbf x_i \in \mathbb R^d$$

## Text Categorization

$$\{(\text{text}_i, y_i) \mid i=1,\ldots, N\}$$
, i.e., inputs are texts.

## Solution

Create a function $m$ that transforms a text into a vector, i.e., $$m: \text{text} \rightarrow \mathbb R^d$$.

$$\mathcal X_d = \{(m(\text{text}_i), y_i) \mid (\text{text}_i, y_i) \in \mathcal X\}$$

## Train the model.

Use $$\mathcal X_d$$ instead of $$\mathcal X$$ to train the model

## Predict

Let $$f$$ be the trained model, and $$t$$ a text. Then the category of $$t$$ can be predicted as: $$f(m(t))$$.

# Text to vector

One of the simplest way to transform a text into a vector is by creating using a Bag of Word (BoW) model.

Let $$\textsf{id}: \text{token} \rightarrow \mathbb N$$, that is a function that given a token returns a unique identifier. For example, $$\textsf{id}(\text{hi})=3$$, $$\textsf{id}(\text{morning})=4$$, and so on. The values $$3$$ and $$4$$ are not important, the important part is that $$3$$ is used to identify token _hi_, and $4$ identifies token _morning_. Using this notation then $$\textsf{id}^{-1}(3)=\text{hi}$$.