import nltk.data
from microtc.utils import save_model

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
