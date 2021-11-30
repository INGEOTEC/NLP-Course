from microtc.utils import tweet_iterator
from os.path import join
import json

path = "snli_1.0"
D = []
for fname in ["snli_1.0_train.jsonl",
              "snli_1.0_dev.jsonl", 
              "snli_1.0_test.jsonl"]:
    d = tweet_iterator(join(path, fname))
    _ = [dict(sentence1=x['sentence1'], sentence2=x['sentence2'],
              klass=x['gold_label']) for x in d if x['gold_label'] != '-']
    D.append(_)

for d, fname in zip(D, ['snli_train.json',
                        'snli_dev.json',
                        'snli_test.json']):
    with open(fname, 'w') as fpt:
        [print(json.dumps(x), file=fpt) for x in d]

