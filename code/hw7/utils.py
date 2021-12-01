from EvoMSA.utils import download
from microtc.utils import load_model
from scipy.sparse import hstack


def transform(data):
    tm = load_model(download('b4msa_En.tm'))
    concat = [[x['sentence1'], x['sentence2']] for x in data]
    return tm.transform(concat)


def transform2(data):
    tm = load_model(download('b4msa_En.tm'))
    sent1 = tm.transform([x['sentence1'] for x in data])
    sent2 = tm.transform([x['sentence2'] for x in data])
    return hstack((sent1, sent2))  