import numpy as np
from scipy.optimize import minimize
from joblib import Parallel, delayed
from tqdm import tqdm
from text_models import Vocabulary
from text_models.utils import date_range
from collections import Counter
from matplotlib import pylab as plt


COUNTRIES = ['MX', 'CO', 'ES', 'AR',
             'PE', 'VE', 'CL', 'EC',
             'GT', 'CU', 'BO', 'DO', 
             'HN', 'PY', 'SV', 'NI', 
             'CR', 'PA', 'UY']
LANG='Es'


# COUNTRIES = ['AU', 'GB', 'CA', 'NZ', 'US']
# LANG='En'


def get_words(date=dict(year=2022, month=1, day=10)):
    """
    >>> words = get_words()
    >>> len(words)
    19
    """

    vocs = Parallel(n_jobs=-1)(delayed(Vocabulary)(date,
                                                   lang=LANG,
                                                   country=country)
                                   for country in tqdm(COUNTRIES))
    words = [{k: v for k, v in voc.voc.items() if not k.count('~')}
             for voc in vocs]
    return words


def zipf_freq(data: dict):
    """
    >>> words = get_words()
    >>> y = zipf_freq(words[0])
    >>> y.shape
    (6586,)
    """

    freq = [f for _, f  in Counter(data).most_common()]
    return np.array(freq)


def zipf(data):
    """
    >>> words = get_words()
    >>> zipf(words[0])
    array([1.75333773e+04, 7.71476911e-01])
    """


def correlation_zipf(data):
    """
    >>> words = get_words()
    >>> correlation_zipf(words[:5])
    array([[ 1.        ,  0.01935657,  0.99308089],
           [ 0.01935657,  1.        , -0.09643028],
           [ 0.99308089, -0.09643028,  1.        ]])
    """

    w = [zipf(d) for d in data]
    tokens = [sum(list(w.values())) for w in data]
    X = np.array([[a, b, c] for (a, b), c in zip(w, tokens)])
    return np.corrcoef(X.T)


def get_words_date_range(init, end):
    """
    >>> init = dict(year=2021, month=11, day=1)
    >>> end = dict(year=2021, month=11, day=3)
    >>> ww = get_words_date_range(init, end)
    >>> len(ww)
    19
    """

    dates = date_range(init, end)
    words = [get_words(d) for d in dates]
    ww = [[w[index] for w in words] for index in range(len(COUNTRIES))]
    return ww


def voc_tokens(data):
    """
    >>> init = dict(year=2021, month=11, day=1)
    >>> end = dict(year=2021, month=11, day=3)
    >>> ww = get_words_date_range(init, end)
    >>> v, n = voc_tokens(ww[0])
    >>> v.shape, n.shape
    ((3,), (3,))
    """
    cnt = Counter(data[0])
    output = [[len(cnt), sum(list(cnt.values()))]]
    for x in data[1:]:
        cnt.update(x)
        _ = [len(cnt), sum(list(cnt.values()))]
        output.append(_)
    output = np.array(output)
    return output[:, 0], output[:, 1]


def heaps(v, n):
    """
    >>> init = dict(year=2021, month=11, day=1)
    >>> end = dict(year=2021, month=11, day=3)
    >>> ww = get_words_date_range(init, end)
    >>> v, n = voc_tokens(ww[0])
    >>> heaps(v, n)
    array([6.5268147 , 0.55190507])
    """


def country_k_beta_max_n():
    init = dict(year=2021, month=11, day=1)
    end = dict(year=2021, month=11, day=30)
    ww = get_words_date_range(init, end)
    lst = []
    for country, w in zip(COUNTRIES, ww):
        v, n = voc_tokens(w)
        k, beta = heaps(v, n)
        lst.append([country, k, beta, n[-1]])
    lst.sort(key=lambda x: x[-1], reverse=True)
    return lst


def plot_alpha_beta():
    init = dict(year=2021, month=11, day=1)
    end = dict(year=2021, month=11, day=30)
    ww = get_words_date_range(init, end)
    beta = [heaps(*voc_tokens(w))[1] for w in ww]
    words = get_words()
    zipf_c = [zipf(d) for d in words]

    # plt.rcParams['text.usetex'] = True
    plt.plot([x for _, x in zipf_c], beta, 'o')
    for y, (_, x), country in zip(beta, zipf_c, COUNTRIES):
        plt.annotate(country, [x, y])
    plt.grid()
    plt.ylabel(r'$\beta$')
    plt.xlabel(r'$\alpha$')
    plt.tight_layout()
    plt.savefig('es_alpha_beta.png', dpi=300)


if __name__ == '__main__':
    plot_alpha_beta()