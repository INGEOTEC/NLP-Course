import numpy as np
from microtc.utils import tweet_iterator
import json

if __name__ == '__main__':
    docs = {x['id']: x['text'] for x in tweet_iterator("dataset.json")}
    P = []
    for model in ['tfidf', 'emoji']:
        y = {x['query']: set(x['knn']) for x in tweet_iterator("{}-sol.json".format(model))}
        hy = {x['query']: x['knn'] for x in tweet_iterator("{}.json".format(model))}
        perf = []
        print("*" * 10, model)
        for k, p in y.items():
            t = set(hy[k])
            perf.append(len(p & t) / len( p | t))
            if k in [0, 10, 20]:
                print("-> ", docs[hy[k][0]])
        perf = np.mean(perf)
        P.append("{:.3f}".format(perf))
    _ = '/autograder/results/results.json'
    # _ = 't.json'
    output = dict(visibility="visible",
                  tests=[dict(name="TF-IDF", 
                         max_score=1, score=P[0]),
                         dict(name="Emoji", 
                         max_score=1, score=P[1])])
    with open(_, 'w') as f:
        json.dump(output, f)        
        