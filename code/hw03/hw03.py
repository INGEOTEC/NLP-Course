import numpy as np


def estimate_bivariate_distribution(string):
    """
    >>> cdn = 'a a b b a a a b'
    >>> estimate_bivariate_distribution(cdn)
    array([[0.42857143, 0.28571429],
           [0.14285714, 0.14285714]])
    """


def compute_conditional_prob(bivariate):
    """
    >>> cdn = 'a a b b a a a b'
    >>> bivariate = estimate_bivariate_distribution(cdn)
    >>> compute_conditional_prob(bivariate)
    array([[0.6, 0.4],
           [0.5, 0.5]])
    """


def generate_sentence(cond_prob, init_index=1, k=3):
    """
    >>> import numpy as np
    >>> np.random.seed(0)
    >>> cdn = 'a a b b a a a b'
    >>> bivariate = estimate_bivariate_distribution(cdn)
    >>> cond = compute_conditional_prob(bivariate)
    >>> generate_sentence(cond, 1, 4)
    [1, 0, 1, 0]
    """

