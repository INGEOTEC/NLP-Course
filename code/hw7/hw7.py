from microtc.utils import tweet_iterator
from sklearn.svm import LinearSVC
from EvoMSA.utils import bootstrap_confidence_interval
import numpy as np
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
from utils import dataset, transform2 as transform
from scipy.sparse import vstack


if __name__ == '__main__':
    train = tweet_iterator('snli_train.json')
    test = tweet_iterator('snli_test.json')
    train = dataset(train)
    test = dataset(test)
    with Pool(cpu_count() - 1) as pool:
        _ = [(i, train[i:i+10000]) for i in range(0, len(train), 10000)]
        Xt = [i for i in tqdm(pool.imap_unordered(transform, _), total=len(_))]
    Xt.sort(key=lambda x: x[0])
    Xt = vstack([x for _, x in Xt])
    m = LinearSVC().fit(Xt, [x['klass'] for x in train])
    hy = m.predict(transform([0, test])[1])
    y = np.array([x['klass'] for x in test])
    print((y == hy).mean())
    print(bootstrap_confidence_interval(y, hy, metric=lambda y, hy: (y == hy).mean()))
