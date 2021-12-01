import json
from microtc.utils import tweet_iterator
from sklearn.svm import LinearSVC
from joblib import Parallel, delayed
from tqdm import tqdm
from utils import transform2 as transform
from scipy.sparse import vstack


if __name__ == '__main__':
    train = list(tweet_iterator('snli_train.json'))
    test = list(tweet_iterator('snli_test.json'))
    _ = [train[i:i+10000] for i in range(0, len(train), 10000)]
    Xt = vstack(Parallel(-1)(delayed(transform)(x) for x in tqdm(_)))
    m = LinearSVC().fit(Xt, [x['klass'] for x in train])
    hy = m.predict(transform(test))
    with open('snli_test.json', 'w') as fpt:
        for d, k in zip(test, hy):
            d['klass'] = k
            print(json.dumps(d), file=fpt)