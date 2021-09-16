from collections import Counter
from typing import List
import re


class NgramLM(object):
    def __init__(self, n: int=2, nlp: object=None) -> None:
        from spacy.lang.en import English
        self._tokens = Counter()
        self._n_grams = Counter()
        self.n = n
        if nlp is None:
            self.nlp = English()

    def process_file(self, fname: str):
        txt = open(fname).read()
        para = [x for x in re.finditer(r"\n\n", txt)]
        index = [0] + [x.end(0) for x in para]
        para = [txt[i:j] for i, j in zip(index, index[1:])]
        self.process_paragraphs([x for x in para if len(x) > 2])

    @staticmethod
    def n_grams(tokens: list, n: int):
        ww = [tokens[i:] for i in range(n)]
        _ = ["~".join(x) for x in zip(*ww)]
        return _

    def tokenize(self, txt: str) -> List[str]:
        _ = [x.norm_.strip() for x in self.nlp(txt)]
        _ = [x for x in _ if len(x)]
        if len(_) == 0:
            return _
        _.insert(0, "<P>")
        _.append("</P>")
        return _

    def process_paragraphs(self, para: List[str]):
        for p in para:
            _ = self.tokenize(p)
            self._tokens.update(_)
            _ = self.n_grams(_, n=self.n)
            self._n_grams.update(_)

    def prob(self, n_gram: str) -> float:
        c_bi = self._n_grams[n_gram]
        a, _ = n_gram.split("~")
        c_token = self._tokens[a]
        return c_bi / c_token

    def sentence_prob(self, txt: str) -> float:
        tokens = self.tokenize(txt)
        ngrams = self.n_grams(tokens, n=self.n)
        p = 1
        for x in ngrams:
            p = p * self.prob(x)
        return p


if __name__ == "__main__":
    from glob import glob
    path = "/Users/mgraffg/software/NLP-Course/code/books/*.txt"
    lm = NgramLM()
    [lm.process_file(fname) for fname in glob(path)]
    pr = lm.sentence_prob("You still have much to learn.")
    print("prob:", pr)
    pr = lm.sentence_prob("Much to learn, you still have.")
    print("prob:", pr)
