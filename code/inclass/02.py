import nltk.data
from spacy.lang.en import English
from microtc.textmodel import TextModel
from microtc.params import OPTION_NONE
from glob import glob
from collections import Counter
from matplotlib import pylab as plt

txt = """This eBook is for the use of anyone anywhere in the United States and
most other parts of the world at no cost and with almost no restrictions
whatsoever. You may copy it, give it away or re-use it under the terms
of the Project Gutenberg License included with this eBook or online at
www.gutenberg.org. If you are not located in the United States, you
will have to check the laws of the country where you are located before
using this eBook."""

sent_detector = nltk.data.load("tokenizers/punkt/english.pickle")

sent_detector.tokenize(txt)

sent_detector.tokenize("I am going to talk to MGG because I want to.")


nlp = English()
nlp.add_pipe("sentencizer")
doc = nlp("This is a sentence. This is another sentence.")
[x for x in doc.sents]

[x for x in nlp(txt).sents]

doc = nlp("This is a sentence. U.S. is another sentence.")
[x for x in doc.sents]

tm = TextModel(num_option=OPTION_NONE, 
               usr_option=OPTION_NONE,
               url_option=OPTION_NONE, 
               emo_option=OPTION_NONE, 
               hashtag_option=OPTION_NONE,
               ent_option=OPTION_NONE, 
               lc=True, del_dup=False, del_punc=False,
               del_diac=False, token_list=[-1])

tm.tokenize("Hello good morning")

heaps = []
counter = Counter()
for fname in glob("../books/*.txt"):
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