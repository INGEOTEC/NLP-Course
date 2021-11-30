from EvoMSA.utils import download
from microtc.utils import load_model
from scipy.sparse import hstack


def dataset(lst):
    output = []
    for x in lst:
        _ = dict(text=[x['sentence1'], x['sentence2']],
                 sentence1=x['sentence1'],
                 sentence2=x['sentence2'],
                 klass=x['klass'])
        output.append(_)
    return output


def transform(data):
    tm = load_model(download('b4msa_En.tm'))
    i, X = data
    return i, tm.transform(X)


def transform2(data):
    tm = load_model(download('b4msa_En.tm'))
    i, X = data
    sent1 = tm.transform([x['sentence1'] for x in X])
    sent2 = tm.transform([x['sentence2'] for x in X])
    return i, hstack((sent1, sent2))