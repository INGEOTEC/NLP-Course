---
layout: default
title: Text Categorization
nav_order: 6
---

# Text Categorization
{: .fs-10 .no_toc }

The **objective** is to 

## Contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---

# Introduction

Text Categorization aims to identify the class of a given text. The approach followed is to treat this problem as a supervised learning problem. As in any supervised learning problem, the starting point is a set of pairs, where the first element of the pair is the element and the second one corresponds to the output, which is the category. 

Let $$\mathcal D = \{(\text{text}_i, y_i) \mid i=1,\ldots, N\}$$ where $$y \in \{c_1, \ldots c_k\}$$ and $$\text{text}_i$$ is a text. $$\mathcal D$$ can be split into two sets $$\mathcal X$$ and $$\mathcal T$$; the former is used to train the algorithm and the later is used to test the algorithms' performance.

We have seen how to use Naive Bayes to solve a classification problem; that is, we start with a set $$\{(\mathbf x_i, y_i) \mid i=1, \ldots, N\}$$ where $$\mathbf x_i \in \mathbb R^d$$, this set is then used to obtain the parameters of a Naive Bayes classifier, and finally, we can use the trained model to test points that have not been used in the training process.

It is important to note that in order to use a classifier such as Naive Bayes, the inputs must have to be represented as vectors, i.e., $\mathbf x \in R^d$. However, in text categorization the inputs are texts.

## Classification

$$\{(\mathbf x_i, y_i) \mid i=1, \ldots, N\}$$ where $$\mathbf x_i \in \mathbb R^d$$

## Text Categorization

$$\{(\text{text}_i, y_i) \mid i=1,\ldots, N\}$$, i.e., inputs are texts.

## Solution

Create a function $m$ that transforms a text into a vector, i.e., $$m: \text{text} \rightarrow \mathbb R^d$$.

$$\mathcal X_d = \{(m(\text{text_i}), y_i) \mid (\text{text_i}, y_i) \in \mathcal X\}$$

## Train the model.

Use $$\mathcal X_d$$ instead of $$\mathcal X$$ to train the model

## Predict

Let $$f$$ be the trained model, and $$t$$ a text. Then the category of $$t$$ can be predicted as: $$f(m(t))$$.

# Text to vector

One of the simplest way to transform a text into a vector is by creating using a Bag of Word (BoW) model.

Let $$\textsf{id}: \text{token} \rightarrow \mathbb N$$, that is a function that given a token returns a unique identifier. For example, $$\textsf{id}(\text{hi})=3$$, $$\textsf{id}(\text{morning})=4$$, and so on. The values $$3$$ and $$4$$ are not important, the important part is that $$3$$ is used to identify token _hi_, and $4$ identifies token _morning_. Using this notation then $$\textsf{id}^{-1}(3)=\text{hi}$$.