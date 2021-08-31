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

book = open("books/11-0.txt").read()

output = sentence_detector.tokenize(book)

uns = open("hw1/tests/books/unsegmented.txt").read()

output = sentence_detector.tokenize(uns)
save_model(output, "sentence-tokenizer.pickle.gz")


nlp = English()
nlp.add_pipe("sentencizer")
doc = nlp("This is a sentence. This is another sentence.")
[x for x in doc.sents]

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
