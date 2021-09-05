from spacy.lang.en import English
from spacy.pipeline.sentencizer import Sentencizer
import numpy as np


class Sent(Sentencizer):
    def predict(self, docs):    
        """Apply the pipe to a batch of docs, without modifying them.

        docs (Iterable[Doc]): The documents to predict.
        RETURNS: The predictions for each document.
        """
        if not any(len(doc) for doc in docs):
            # Handle cases where there are no tokens in any docs.
            guesses = [[] for doc in docs]
            return guesses
        guesses = []
        for doc in docs:
            doc_guesses = [False] * len(doc)
            if len(doc) > 0:
                start = 0
                seen_period = False
                doc_guesses[0] = True
                for i, token in enumerate(doc):
                    is_in_punct_chars = token.text in self.punct_chars
                    if not is_in_punct_chars:
                        is_in_punct_chars = "." in token.text
                    print("Doing token:", token, is_in_punct_chars)
                    if seen_period and not token.is_punct and not is_in_punct_chars:
                        doc_guesses[start] = True
                        start = token.i
                        seen_period = False
                    elif is_in_punct_chars:
                        seen_period = True
                if start < len(doc):
                    doc_guesses[start] = True
            guesses.append(doc_guesses)
        return guesses    

nlp = English()
sent = Sent()
doc = nlp("I am living in the U.S. I work in ...")
sent(doc)
len(list(doc.sents))

[(x, x.is_sent_start) for x in doc]
# [(I, True), (am, False), (living, False), 
#  (in, False), (the, False), (U.S., False), 
#  (I, False), (work, False), (in, False), 
#  (..., False)]

[x.is_sent_start for x in doc]
# [True, False, False, False, False, 
#  False, False, False, False, False]

# [True, False, False, False, False, 
#  False, True, False, False, False]

hy = [True, False, False, False, False, 
      False, False, False, False, False]

y =  [True, False, False, False, False, 
      False, True, False, False, False]

y = np.array(y)
(y == hy).mean()