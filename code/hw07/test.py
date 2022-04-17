from microtc.utils import tweet_iterator
import numpy as np
import json

if __name__ == '__main__':
    G = list(tweet_iterator("analogy_gold.json"))
    output = list(tweet_iterator("analogy_predictions.json"))
    perf = np.mean([y['d'] in hy['d'] for y, hy in zip(G[10:], output)])
    _ = '/autograder/results/results.json'
    # _ = 't.json'
    print(output[2])
    output = dict(visibility="visible",
                  tests=[dict(name="Accuracy", 
                         max_score=1, score="{:.3f}".format(perf))])
    with open(_, 'w') as f:
        json.dump(output, f)        
        