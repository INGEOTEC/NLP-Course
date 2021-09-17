from collections import Counter
from typing import List
import re
import random


class NgramLM(object):
    """N-Gram Language Model

    :param n: Size of the n-gram, e.g., n=2 bigrams
    :type n: int
    :param nlp: Tokenizer, default spacy.lang.en.English
    """

    def __init__(self, n: int=2, nlp: object=None) -> None:
        from spacy.lang.en import English
        self._tokens = Counter()
        self._n_grams = Counter()
        self.n = n
        if nlp is None:
            self.nlp = English()

    def process_paragraphs(self, para: List[str]):
        """
        >>> ngram = NgramLM()
        >>> ngram.process_paragraphs(["xxx xyx xxy", "xxy aaa baa"])
        >>> len(ngram._tokens)
        7
        >>> sum(ngram._tokens.values())
        10
        >>> len(ngram._n_grams)
        8
        """
        for p in para:
            _ = self.tokenize(p)
            self._tokens.update(_)
            _ = self.n_grams(_)
            self._n_grams.update(_)    

    def process_file(self, fname: str):
        txt = open(fname).read()
        para = [x for x in re.finditer(r"\n\n", txt)]
        index = [0] + [x.end(0) for x in para]
        para = [txt[i:j] for i, j in zip(index, index[1:])]
        self.process_paragraphs([x for x in para if len(x) > 2])

    def tokenize(self, txt: str, markers: bool=True) -> List[str]:
        """Tokenize a text
        
        :param txt: Text
        :type txt: str
        :param markers: include starting and ending markers
        :type markers: bool

        >>> ngram = NgramLM(n=2)
        >>> ngram.tokenize("Good morning!")
        ['<P>', 'good', 'morning', '!', '</P>']
        >>> ngram.tokenize("Good morning!", markers=False)
        ['good', 'morning', '!']
        """

        _ = [x.norm_.strip() for x in self.nlp(txt)]
        _ = [x for x in _ if len(x)]
        if len(_) == 0:
            return _
        if markers:
            _.insert(0, "<P>")
            _.append("</P>")
        return _

    def n_grams(self, tokens: list, n: int=None):
        """Create n-grams from a list of tokens
        
        :param tokens: List of tokens
        :param n: Size of the n-gram
        :type n: int

        >>> ngram = NgramLM(n=3)
        >>> tokens = ngram.tokenize("Good morning!")
        >>> ngram.n_grams(tokens)
        ['<P>~good~morning', 'good~morning~!', 'morning~!~</P>']
        """
        n = self.n if n is None else n
        ww = [tokens[i:] for i in range(n)]
        _ = ["~".join(x) for x in zip(*ww)]
        return _

    def inv_n_grams(self, txt: str):
        """Inverse of n_grams, from the string representation
        of an n-gram computes the tokens
        
        :param txt: string representation of n-gram

        >>> ngram = NgramLM()
        >>> ngram.inv_n_grams('good~morning')
        ['good', 'morning']
        """

        return txt.split("~")

    def prob(self, n_gram: str) -> float:
        """Probability P(w_n | w_{1:n-1}) where the string
        is represented as n-grams

        :param n_gram: string representation of an n-gram

        >>> ngram = NgramLM()
        >>> ngram.process_paragraphs(["xxx xyx xxy", "xxy aaa baa"])
        >>> ngram.prob("xxy~aaa")
        0.5
        >>> ngram.prob("<P>~aaa")
        0.0
        """

        c_bi = self._n_grams[n_gram]
        a, _ = self.inv_n_grams(n_gram)
        c_token = self._tokens[a]
        return c_bi / c_token

    def sentence_prob(self, txt: str, markers: bool=False) -> float:
        """Probability of a sentence P(w_1, w_2, ..., w_n)
        
        :param txt: text
        :param markers: include starting and ending markers
        :type markers: bool        

        >>> ngram = NgramLM()
        >>> ngram.process_paragraphs(["xxx xyx xxy", "xyx aaa xxx"])
        >>> ngram.sentence_prob("xxx xyx aaa")
        0.25
        """

        tokens = self.tokenize(txt, markers=markers)
        ngrams = self.n_grams(tokens)
        p = 1
        for x in ngrams:
            _ = self.prob(x)
            p = p * _
        return p

    def log_sentence_prob(self, txt: str, markers: bool=True) -> float:
        pass

    def generate_sentence(self, prev: str=None, n_tokens: int=10, random_size: int=5) -> List[str]:
        """
        Generate a sentence starting the text (prev)

        :param prev: start of the sentence as a string
        :type prev: str
        :param n_tokens: Number of tokens to generate
        :type n_tokens: int

        >>> ngram = NgramLM()
        >>> ngram.process_paragraphs(["xxx xyx xxy", "xyx aaa xxx"])
        >>> ngram.generate_sentence(prev="aaa", n_tokens=1)
        ['aaa', 'xxx']
        """

        if prev is None:
            ll =  self.tokenize("-")
            prev = ll[:len(ll) // 2]
        else:
            prev = self.tokenize(prev, markers=False)
        n = self.n
        for _ in range(n_tokens):
            _ = prev[-(self.n - 1):]
            ngrams = [self.n_grams(_ + [x])[0]  for x in self._tokens.keys()]
            count_ngrams = [[x, self._n_grams.get(x, 0)] for x in ngrams]
            count_ngrams = [x for x in count_ngrams if x[1] > 0]
            if len(count_ngrams) == 0:
                break
            count_ngrams.sort(key=lambda x: x[1], reverse=True)
            cnt = min(len(count_ngrams), random_size)
            x = count_ngrams[random.randrange(cnt)][0]
            prev.append(self.inv_n_grams(x)[-1])
        return prev