import json
from microtc.utils import tweet_iterator
from sklearn.svm import LinearSVC
import numpy as np
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
from utils import transform as transform
from scipy.sparse import vstack


if __name__ == '__main__':
    train = list(tweet_iterator('snli_train.json'))
    test = list(tweet_iterator('snli_test.json'))
    with Pool(cpu_count() - 1) as pool:
        _ = [(i, train[i:i+10000]) for i in range(0, len(train), 10000)]
        Xt = [i for i in tqdm(pool.imap_unordered(transform, _), total=len(_))]
    Xt.sort(key=lambda x: x[0])
    Xt = vstack([x for _, x in Xt])
    m = LinearSVC().fit(Xt, [x['klass'] for x in train])
    hy = m.predict(transform([0, test])[1])
    with open('snli_test.json', 'w') as fpt:
        for d, k in zip(test, hy):
            d['klass'] = k
            print(json.dumps(d), file=fpt)