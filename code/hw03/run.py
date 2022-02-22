import json
import doctest
import hw03
import numpy as np
np.set_printoptions(formatter={'float': lambda x: "{:0.4f}".format(x)})


if __name__ == '__main__':
    output = {
        "score": 0,
        "visibility": "visible",
        "stdout_visibility": "visible"
    }
    fname = '/autograder/results/results.json'
    # fname = 't.json'
    # doctest.testmod(hw03)
    print("*" * 10)
    bivariate = hw03.estimate_bivariate_distribution(open('seq2.txt').read())
    print(str(bivariate))
    print("*" * 10)
    cond = hw03.compute_conditional_prob(bivariate)
    print(str(cond))
    print("*" * 10)
    sentence = hw03.generate_sentence(cond, k=10)
    print(sentence)
    print("*" * 10)
    with open(fname, 'w') as f:
        json.dump(output, f)
