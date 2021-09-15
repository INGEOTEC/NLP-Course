import numpy as np
from spacy.pipeline.sentencizer import Sentencizer
from glob import glob
from spacy.lang.en import English


def metrics(a, b):
    from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score
    return (accuracy_score(a, b),
            recall_score(a, b),
            precision_score(a, b),
            f1_score(a, b))


def performance(colgate=None):
    colgate = colgate if colgate is not None else Sentencizer()
    nlp = English()
    # colgate = ColgateSBD()

    output = []
    for test in glob("marked-*.txt"):
        input = test.replace("marked-", "") 
        txt = open(input).read()
        tokens = nlp(open(test).read())
        hy_tokens = colgate(nlp(txt))
        assert len(tokens) == len(hy_tokens)
        _ = [b.is_punct and a.text == "#"
             for a, b in zip(tokens, hy_tokens)]
        y = np.empty_like(_, dtype=bool)
        y[1:] = _[:-1]
        y[0] = True
        hy = np.array([x.is_sent_start for x in hy_tokens])
        _ = metrics(y, hy)
        output.append((test, _, y.sum()))
    return output


if __name__ == "__main__":
    from hw2 import ColgateSBD
    from glob import glob
    from spacy.lang.en import English

    output = performance(ColgateSBD())
    for input, perf, n_sent in output:
        print("Input:", input, perf, "Number of sentences:", n_sent)


