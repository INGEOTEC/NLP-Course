import unittest
from sklearn.metrics import recall_score
from microtc.utils import tweet_iterator
import json

if __name__ == '__main__':
    y = [x['klass'] for x in tweet_iterator("semeval2017_En_test.json")]
    hy = [x['klass'] for x in tweet_iterator("predictions.json")]
    perf = recall_score(y, hy, average="macro")
    _ = '/autograder/results/results.json'
    # _ = 't.json'
    output = dict(visibility="visible",
                  tests=[dict(name="macro-Recall", 
                         max_score=1, score="{:.3f}".format(perf))])
    with open(_, 'w') as f:
        json.dump(output, f)        
        