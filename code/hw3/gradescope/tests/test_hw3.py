from hw3 import NgramLM
import unittest
from gradescope_utils.autograder_utils.decorators import weight, number
import numpy as np


class TestNgramLM(unittest.TestCase):    
    @number("1.1")
    @weight(10/6)
    def test_bigram_prob(self):
        "Test P(w_2 | w_1)"
        ngram = NgramLM(n=2)
        ngram.process_paragraphs(["xxx xyx xxy",
                                  "xyx aaa xxx"])
        tokens = ngram.tokenize("xyx xxy", markers=False)
        n_grams = ngram.n_grams(tokens)
        assert len(n_grams) == 1
        p = ngram.prob(n_grams[0])
        self.assertEqual(p, 0.5)

    @number("1.2")
    @weight(10/6)
    def test_prob(self):
        "Test P(w_2 | w_1, w_2)"
        ngram = NgramLM(n=3)
        ngram.process_paragraphs(["xxx xyx xxy",
                                  "xxx xyx aaa",
                                  "xyx aaa xxx"])
        tokens = ngram.tokenize("xxx xyx xxy", markers=False)
        n_grams = ngram.n_grams(tokens)
        self.assertEqual(len(n_grams), 1)
        p = ngram.prob(n_grams[0])
        self.assertEqual(p, 0.5)
        tokens = ngram.tokenize("xxy xyx aaa", markers=False)
        n_grams = ngram.n_grams(tokens)
        try:
            p = ngram.prob(n_grams[-1])
        except ZeroDivisionError:
            return
        self.assertEqual(p, 0.0)

    @number("1.3")
    @weight(10/6)
    def test_bigram_log_sentence_prob(self):
        "Test P(w_1, w_2, w_3)"
        ngram = NgramLM(n=2)
        ngram.process_paragraphs(["xxx xyx xxy",
                                  "xyx aaa xxx"])
        pr = ngram.sentence_prob("xxx xyx aaa")
        pr_log = ngram.log_sentence_prob("xxx xyx aaa")
        self.assertAlmostEqual(pr, np.exp(pr_log))

    @number("1.4")
    @weight(10/6)
    def test_log_sentence_prob(self):
        "Test P(w_1, w_2, w_3, w_4) on 3-grams"
        ngram = NgramLM(n=3)
        ngram.process_paragraphs(["xxx xyx xxy xxa",
                                  "xxx xyx xxy xxc",
                                  "xyx xxy xxb xxx"])
        pr = ngram.sentence_prob("xxx xyx xxy xxb")
        pr_log = ngram.log_sentence_prob("xxx xyx xxy xxb")
        self.assertAlmostEqual(pr, np.exp(pr_log))
        self.assertAlmostEqual(np.exp(pr_log), 1 / 3)

    @number("2.1")
    @weight(10/6)
    def test_bigram_generate_sentence(self):
        "Generate a sentece using a bigram Language Model"
        ngram = NgramLM(n=2)
        ngram.process_paragraphs(["xxx xyx xxy xxa",
                                  "xxx xyx xxy xxc",
                                  "xyx xxy xxa xxx"])
        sent = ngram.generate_sentence(prev="xyx", n_tokens=2)
        self.assertEqual(len(sent), 3)

    @number("2.2")
    @weight(10/6)
    def test_generate_sentence(self):
        "Generate a sentece using a 3-gram Language Model"
        ngram = NgramLM(n=3)
        ngram.process_paragraphs(["xxx xyx xxy xxa",
                                  "xxx xyx xxy xxc",
                                  "xyx xxy xxa xxx"])
        sent = ngram.generate_sentence(prev="xyx xxy", n_tokens=2)
        self.assertEqual(len(sent), 4)
        with self.assertRaises(IndexError):
            ngram.generate_sentence(prev="xxy")