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
from scipy.optimize import minimize, minimize_scalar
from matplotlib import pylab as plt
from nltk.stem.porter import PorterStemmer


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

heaps = []
counter = Counter()
for fname in glob("books/*.txt"):
    txt = open(fname).read()
    tokens = tm.tokenize(txt)
    counter.update(tokens)
    V = len(counter)
    N = sum([v for v in counter.values()])
    heaps.append([N, V])

plt.plot([x for x, _ in heaps], [x for _, x in heaps])
plt.grid()
plt.xlabel("N")
plt.ylabel("|V|")
plt.tight_layout()
plt.savefig("heaps-law.png", dpi=300) 


def error_heaps(beta):
    y = [y for _, y in heaps]
    x = np.array([x for x, _ in heaps])
    hy = x**beta
    return ((y - hy)**2).sum()


res = minimize(error_heaps, [0.7], method='nelder-mead',
               options={'xatol': 1e-8, 'disp': True})

xx = minimize_scalar(error_heaps, bounds=(0, 1), method='bounded')               


plt.plot([x for x, _ in heaps], [x for _, x in heaps])
x = np.array([x for x, _ in heaps])
hy = x**res.x[0]
plt.plot(x, hy)


stemmer = PorterStemmer()
stemmer.stem("playing")