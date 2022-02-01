import unittest
import json
import doctest
import hw01


if __name__ == '__main__':
    output = {
        "score": 0,
        "visibility": "visible",
        "stdout_visibility": "visible"
    }
    fname = '/autograder/results/results.json'
    # fname = 't.json'
    doctest.testmod(hw01)
    with open(fname, 'w') as f:
        json.dump(output, f)
