import json
import doctest
import hw04
from hw04 import ngrams_frequency, compute_conditional_prob
import numpy as np
import doctest


def test_code(arg):
  doctest.run_docstring_examples(arg, globals())


if __name__ == '__main__':
    output = {
        "score": 0,
        "visibility": "visible",
        "stdout_visibility": "visible"
    }
    fname = '/autograder/results/results.json'
    # fname = 't.json'
    # lst = ['I enjoy', 'I like to play']
    # ngrams = hw03.ngrams_frequency(lst, n=3)
    # ngrams.most_common(1)    
    print("*" * 10)
    test_code(hw04.ngrams_frequency)
    print("*" * 10)
    test_code(hw04.compute_conditional_prob)
    print("*" * 10)
    print(hw04.main())
    print("*" * 10)
    with open(fname, 'w') as f:
        json.dump(output, f)
