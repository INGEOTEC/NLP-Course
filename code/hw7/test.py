from microtc.utils import tweet_iterator
import numpy as np
import json
from EvoMSA.utils import bootstrap_confidence_interval

if __name__ == '__main__':
    G = tweet_iterator("snli_gold.json")
    y = np.array([x['klass'] for x in G])
    output = tweet_iterator("snli_test.json")
    hy = np.array([x['klass'] for x in output])
    ci = bootstrap_confidence_interval(y, hy, metric=lambda y, hy: (y == hy).mean())
    print(ci)
    perf = (y == hy).mean()
    _ = '/autograder/results/results.json'
    # _ = 't.json'
    output = dict(visibility="visible",
                  tests=[dict(name="Accuracy", 
                         max_score=1, score="{:.3f}".format(perf))])
    with open(_, 'w') as f:
        json.dump(output, f)        
        