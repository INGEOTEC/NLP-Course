from microtc.utils import tweet_iterator, load_model
from b4msa.textmodel import TextModel
from sklearn.svm import LinearSVC
from EvoMSA.utils import download
import numpy as np
from multiprocessing import Pool, cpu_count
from tqdm import tqdm


train = tweet_iterator('snli_train.json')
test = tweet_iterator('snli_test.json')

def dataset(lst):
    output = []
    for x in lst:
        _ = dict(text=[x['sentence1'], x['sentence2']],
                 klass=x['klass'])
        output.append(_)
    return output

def transform(data):
    tm = load_model(download('b4msa_En.tm'))
    i, data = data
    return i, tm.transform(data)

train = dataset(train)
test = dataset(test)
# tm = TextModel(lang="english").fit(train)
tm = load_model(download('b4msa_En.tm'))

with Pool(cpu_count() - 1) as pool:
    _ = [(i, train[i:i+10000]) for i in range(0, len(train), 10000)]
    Xt = [i for i in tqdm(pool.imap_unordered(transform, _), total=len(_))]

Xt = tm.transform(train)

m = LinearSVC().fit(Xt, [x['klass'] for x in train])
hy = m.predict(tm.transform(test))
y = np.array([x['klass'] for x in test])
(y == hy).mean()
