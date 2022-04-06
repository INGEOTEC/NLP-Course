import nltk.data
from microtc.utils import save_model
from spacy.lang.en import English

txt = """This eBook is for the use of anyone anywhere in the United States and
most other parts of the world at no cost and with almost no restrictions
whatsoever. You may copy it, give it away or re-use it under the terms
of the Project Gutenberg License included with this eBook or online at
www.gutenberg.org. If you are not located in the United States, you
will have to check the laws of the country where you are located before
using this eBook."""

sentence_detector = nltk.data.load("tokenizers/punkt/english.pickle")

sentence_detector.tokenize(txt)

# book = open("books/11-0.txt").read()

# output = sentence_detector.tokenize(book)

# uns = open("hw1/tests/books/unsegmented.txt").read()

# output = sentence_detector.tokenize(uns)
# save_model(output, "sentence-tokenizer.pickle.gz")


nlp = English()
nlp.add_pipe("sentencizer")
doc = nlp("This is a sentence. This is another sentence.")
[x for x in doc.sents]

sent1, sent2 = list(doc.sents)
type(sent1)

print(len(sent1))
# 5
[x.text for x in sent1]
# ['This', 'is', 'a', 'sentence', '.']

sent1[-1].is_punct
# True
type(sent1[-1])
# spacy.tokens.token.Token

# Tokenizer

nlp.tokenizer.explain("This is a sentence.")
# [('TOKEN', 'This'),
#  ('TOKEN', 'is'),
#  ('TOKEN', 'a'),
#  ('TOKEN', 'sentence'),
#  ('SUFFIX', '.')]

nlp.tokenizer.explain("This is US.")
# [... ('TOKEN', 'US'), ('SUFFIX', '.')]
nlp.tokenizer.explain("This is U.S.")
# [... ('TOKEN', 'is'), ('TOKEN', 'U.S.')]
nlp.tokenizer.explain("I don't want to ...")
# [('TOKEN', 'I'),
#  ('SPECIAL-1', 'do'),
#  ('SPECIAL-2', "n't"),
#  ('TOKEN', 'want'),
#  ('TOKEN', 'to'),
#  ('PREFIX', '...')]

nlp.tokenizer.explain("This is U.S.")

nlp = English()
nlp.add_pipe("sentencizer")
doc = nlp("I live in the U.S. with my family.")
len(list(doc.sents))

doc = nlp("I live in the U.S. I like to ...")
len(list(doc.sents))

doc = nlp("I am living in the U.S. I work in ...")
len(list(doc.sents))


[x.norm_ for x in nlp("I don't want to ...")]
# ['i', 'do', 'not', 'want', 'to', '...']

_ = nlp("I am going to talk to M.G.G. because I want to")
len([x for x in _.sents])

[x for x in nlp(txt).sents]

txt = """And he looked over at the alarm clock, ticking on the chest of drawers.
“God in Heaven!” he thought. It was half past six and the hands were
quietly moving forwards, it was even later than half past, more like
quarter to seven. Had the alarm clock not rung? He could see from the
bed that it had been set for four o’clock as it should have been; it
certainly must have rung."""

[x for x in nlp(txt).sents]

sentence_detector.tokenize("I am living in the U.S. I work in ...")

pp = sentence_detector._params
pp.abbrev_types
pp.collocations
pp.sent_starters
pp.ortho_context

sentence_detector.tokenize("I am living in the Mex. with my ...")

