from microtc.utils import tweet_iterator, load_model
from b4msa.textmodel import TextModel
from sklearn.svm import LinearSVC
from EvoMSA.utils import download

train = tweet_iterator('snli_train.json')
test = tweet_iterator('snli_test.json')

def dataset(lst):
    output = []
    for x in lst:
        _ = dict(text=[x['sentence1'], x['sentence2']],
                 klass=x['klass'])
        output.append(_)
    return output

train = dataset(train)
test = dataset(test)
# tm = TextModel(lang="english").fit(train)
tm = load_model(download('b4msa_En.tm'))
Xt = tm.transform(train)

m = LinearSVC().fit(Xt, [x['klass'] for x in train])
hy = m.predict(tm.transform(test))
