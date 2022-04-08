import numpy as np
from wordcloud import WordCloud as WC
from matplotlib import pylab as plt
from b4msa.textmodel import TextModel
from microtc.params import OPTION_GROUP, OPTION_DELETE, OPTION_NONE
from b4msa.lang_dependency import LangDependency
from nltk.stem.porter import PorterStemmer
import re
from microtc.textmodel import SKIP_SYMBOLS
import unicodedata
# %pylab inline

## Users
text = 'Hi @xx, @mm is talking about you.'
re.sub(r"@\S+", "", text)

text = 'Hi @xx, @mm is talking about you.'
re.sub(r"@\S+", "_usr", text)

## URL
text = "go http://google.com, and find out"
re.sub(r"https?://\S+", "", text)

## Numbers
text = "we have won 10 M"
re.sub(r"(\d+\.\d+)|(\.\d+)|(\d+\.)|(\d+)", "_num", text)

## Case sensitive
text = "Mexico"
text.lower()

## Punctuation
text = "Hi! good morning,"
output = ""
for x in text:
    if x in SKIP_SYMBOLS:
        continue
    output += x
output

## Diacritic
text = 'México'
output = ""
for x in unicodedata.normalize('NFD', text):
    o = ord(x)
    if 0x300 <= o and o <= 0x036F:
        continue
    output += x
output

## Stop words
lang = LangDependency('english')

text = 'Good morning! Today, we have a warm weather.'
output = []
for word in text.split():
    if word.lower() in lang.stopwords[len(word)]:
        continue
    output.append(word)
output = " ".join(output) 


## Stemmming and Lemmatization
stemmer = PorterStemmer()

text = 'I like playing football'
output = []
for word in text.split():
    w = stemmer.stem(word)
    output.append(w)
output = " ".join(output) 
output

text = 'I like playing football on Saturday'
words = text.split()
n = 3
n_grams = []
for a in zip(*[words[i:] for i in range(n)]):
    n_grams.append("~".join(a))
n_grams


text = 'I like playing'
q = 4
q_grams = []
for a in zip(*[text[i:] for i in range(q)]):
    q_grams.append("".join(a))
q_grams

text = 'I like playing football with @mgraffg'
tm = TextModel(token_list=[-1, 5], lang='english', 
               usr_option=OPTION_GROUP,
               stemming=True)
tm.text_transformations(text)

tm.tokenize(text)