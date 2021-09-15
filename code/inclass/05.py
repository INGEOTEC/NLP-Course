from spacy.lang.en import English
from glob import glob
import re
from tqdm import tqdm
from collections import Counter
from typing import List
import random


def n_grams(tokens: list, n: int):
    ww = [tokens[i:] for i in range(n)]
    _ = ["~".join(x) for x in zip(*ww)]
    return _


nlp = English()
# nlp.add_pipe("sentencizer")

path = "/Users/mgraffg/software/NLP-Course/code/books/*.txt"
tokens = []

## SHOW tqdm
for fname in tqdm(glob(path)):
    txt = open(fname).read()
    para = [x for x in re.finditer(r"\n\n", txt)]
    # break
    ## SHOW 
    index = [0] + [x.end(0) for x in para]
    # break
    ## SHOW
    para = [txt[i:j] for i, j in zip(index, index[1:])]
    # break
    ## SHOW
    para = [x for x in para if len(x) > 2]
    for p in para:
        _ = [x.norm_.strip() for x in nlp(p)]
        _ = [x for x in _ if len(x)]
        if len(_) == 0:
            continue
        _.insert(0, "<P>")
        _.append("</P>")
        # break
        ## SHOW
        tokens.append(_)

bigrams = [n_grams(x, 2) for x in tokens]    
# Count tokens
count_tokens = Counter()
[count_tokens.update(x) for x in tokens]
count_tokens.most_common(10)
## SHOW

# Count bigrams
count_bigrams = Counter()
[count_bigrams.update(x) for x in bigrams]
count_bigrams.most_common(10)
## SHOW


def prob(bigram):
    c_bi = count_bigrams[bigram]
    a, b = bigram.split("~")
    c_token = count_tokens[a]
    return c_bi / c_token

prob("of~the")
## SHOW

txt = "you still have much to learn."
sentence = [x.norm_.rstrip() for x in nlp(txt)]
sentence.insert(0, "<P>")
sentence.append("</P>")
## SHOW
sent_bi = n_grams(sentence, 2)
## SHOW

output = 1
for x in sent_bi:
    print("prob", x, prob(x))
    ## compute prob
    output = output * prob(x)
output    

txt = "Much to learn, you still have."
sentence = [x.norm_.rstrip() for x in nlp(txt)]
sentence.insert(0, "<P>")
sentence.append("</P>")
## SHOW
sent_bi = n_grams(sentence, 2)

output = 1
for x in sent_bi:
    print("prob", prob(x))
    ## compute prob
    output = output * prob(x)
output


## Organize the code

class NgramLM(object):
    def __init__(self, n: int=2) -> None:
        self._tokens = Counter()
        self._n_grams = Counter()
        self.n = n

    def process_file(self, fname: str):
        txt = open(fname).read()
        para = [x for x in re.finditer(r"\n\n", txt)]
        # break
        ## SHOW 
        index = [0] + [x.end(0) for x in para]
        # break
        ## SHOW
        para = [txt[i:j] for i, j in zip(index, index[1:])]
        self.process_paragraphs([x for x in para if len(x) > 2])

    @staticmethod
    def tokenize(txt: str) -> List[str]:
        _ = [x.norm_.strip() for x in nlp(txt)]
        _ = [x for x in _ if len(x)]
        if len(_) == 0:
            return _
        _.insert(0, "<P>")
        _.append("</P>")
        return _

    def process_paragraphs(self, para: List[str]):
        # break
        ## SHOW
        for p in para:
            _ = self.tokenize(p)
            # break
            ## SHOW
            self._tokens.update(_)
            _ = n_grams(_, n=self.n)
            self._n_grams.update(_)

    def prob(self, n_gram: str) -> float:
        c_bi = self._n_grams[n_gram]
        a, _ = n_gram.split("~")
        c_token = self._tokens[a]
        return c_bi / c_token

    def sentence_prob(self, txt: str) -> float:
        tokens = self.tokenize(txt)
        ngrams = n_grams(tokens, n=self.n)
        p = 1
        for x in ngrams:
            p = p * self.prob(x)
        return p


lm = NgramLM()
for fname in glob(path):
    lm.process_file(fname)
    print("prob:", lm.prob("of~the"))
## SHOW

lm.sentence_prob("You still have much to learn.")
lm.sentence_prob("Much to learn, you still have.")

# Generate a sentence

prev = "she"
for i in range(10):
    ngrams = [n_grams([prev, x], n=lm.n)[0]  for x in lm._tokens.keys()]
    count_ngrams = [[x, lm._n_grams.get(x, 0)] for x in ngrams]
    count_ngrams.sort(key=lambda x: x[1], reverse=True)

    xx = count_ngrams[:5][random.randrange(5)]
    prev = xx[0].split("~")[1]
    print(prev)
