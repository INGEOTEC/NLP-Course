# Copyright 2021 Mario Graff Guerrero

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from microtc.textmodel import TextModel
from microtc.params import OPTION_NONE
from glob import glob
from collections import Counter
import numpy as np
from scipy.optimize import minimize
from matplotlib import pylab as plt
from nltk.stem.porter import PorterStemmer
from typing import Callable, Iterable


tm = TextModel(num_option=OPTION_NONE, 
               usr_option=OPTION_NONE,
               url_option=OPTION_NONE, 
               emo_option=OPTION_NONE, 
               hashtag_option=OPTION_NONE,
               ent_option=OPTION_NONE, 
               lc=False, del_dup=False, del_punc=False,
               del_diac=False, token_list=[-1])

tm.tokenize("Hello good morning")

# Count the number of words

def N_tokens_types(fname: str, 
                   counter: Counter,
                   tm: Callable[[str], Iterable[str]]):
    txt = open(fname).read()
    tokens = tm(txt)
    counter.update(tokens)    
    N = sum([v for v in counter.values()])
    return N, len(counter)


counter = Counter()
heaps = [N_tokens_types(fname, counter, tm.tokenize)
         for fname in glob("books/*.txt")]


plt.plot([x for x, _ in heaps], [x for _, x in heaps], '*')
plt.grid()
plt.xlabel("N")
plt.ylabel("|V|")
plt.tight_layout()
plt.savefig("heaps-law2.png", dpi=300) 

def error(coef):
    y = [y for _, y in heaps]
    x = np.array([x for x, _ in heaps])
    hy = coef[0] * x**coef[1]
    return ((y - hy)**2).sum()


res = minimize(error, [1, 0.7], 
               method='nelder-mead',
               options={'disp': True})


plt.plot([x for x, _ in heaps], [x for _, x in heaps], '.')
plt.grid()
x = np.array([x for x, _ in heaps])
hy = res.x[0] * x**res.x[1]
plt.plot(x, hy)
plt.xlabel("N")
plt.ylabel("|V|")
plt.tight_layout()
plt.savefig("heaps-law3.png", dpi=300) 


stemmer = PorterStemmer()
stemmer.stem("playing")

## Using another tokenizer

tm = TextModel(num_option=OPTION_NONE, 
               usr_option=OPTION_NONE,
               url_option=OPTION_NONE, 
               emo_option=OPTION_NONE, 
               hashtag_option=OPTION_NONE,
               ent_option=OPTION_NONE, 
               lc=True, del_dup=False, del_punc=False,
               del_diac=False, token_list=[-1])

counter = Counter()
heaps = [N_tokens_types(fname, counter, tm.tokenize)
         for fname in glob("books/*.txt")]

res = minimize(error, [1, 0.7], 
               method='nelder-mead',
               options={'disp': True})

res.x


def n_grams(words: list, n: int):
    ww = [words[i:] for i in range(n)]
    _ = ["~".join(x) for x in zip(*ww)]
    return _

words = ['a', 'b', 'c', 'd']
n_grams(words, 2)
# ['a~b', 'b~c', 'c~d']
n_grams(words, 3)
# ['a~b~c', 'b~c~d']
n_grams(words, 4)
# ['a~b~c~d']