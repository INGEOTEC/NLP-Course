from microtc.utils import tweet_iterator, load_model
from sklearn.metrics import recall_score
from glob import glob
import numpy as np
import json
import os
from EvoMSA.utils import LabelEncoderWrapper


if __name__ == '__main__':
    fname = '/autograder/results/results.json'
    # fname = 't.json'
    perf = load_model('performance.gz')
    better = []
    for test in glob('*test.json'):
      y = [x['klass'] for x in tweet_iterator(test)]
      le = LabelEncoderWrapper().fit(y)
      y = le.transform(y)
      output = "hy/" + test
      if not os.path.isfile(output):
        better.append(False)
      else:
        hy = le.transform([x['klass'] for x in tweet_iterator(output)])
        p = recall_score(y, hy, average='macro')
        better.append(perf[test] < p)
      print(test, better[-1])
    print("Better {:0.4}".format(np.mean(better)))
    output = {
        "score": 0,
        "visibility": "visible",
        "stdout_visibility": "visible"
    }    
    with open(fname, 'w') as f:
        json.dump(output, f)
