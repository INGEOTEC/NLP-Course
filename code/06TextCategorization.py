import numpy as np
from b4msa.textmodel import TextModel
from EvoMSA.tests.test_base import TWEETS
from scipy.stats import norm
from matplotlib import pylab as plt
# %pylab inline

plt.rcParams['text.usetex'] = True

pos = norm(loc=3, scale=2.5)
neg = norm(loc=-0.5, scale=0.75)
plt.clf()
for ins, color in zip([pos, neg], ['b', 'r']):
    x = np.linspace(ins.ppf(0.001),
                    ins.ppf(0.999))
    plt.plot(x, ins.pdf(x), color)
plt.legend(['Positive', 'Negative'])
plt.grid()
plt.tight_layout()
plt.savefig('two_normals.png', dpi=300)

_min = neg.ppf(0.05)
_max = neg.ppf(0.95)
D = [(x, 0) for x in neg.rvs(100) if x >= _min and x <= _max]
D += [(x, 1) for x in pos.rvs(1000) if x < _min or x > _max]

plt.clf()
plt.plot([x for x, k in D if k==1],
         pos.pdf([x for x, k in D if k==1]), 'b.')
plt.plot([x for x, k in D if k==0],
         neg.pdf([x for x, k in D if k==0]), 'r.')
plt.grid()
plt.legend(['Positive', 'Negative'])
plt.tight_layout()
plt.savefig('two_normal_samples.png', dpi=300)

l_pos = norm(loc=np.mean([x for x, k in D if k == 1]),
             scale=np.std([x for x, k in D if k == 1]))
l_neg = norm(loc=np.mean([x for x, k in D if k == 0]),
             scale=np.std([x for x, k in D if k == 0]))            

_, priors = np.unique([k for _, k in D], return_counts=True)
N = priors.sum()
prior_pos = priors[1] / N
prior_neg = priors[0] / N

x = np.array([x for x, _ in D])
x.sort()
post_pos = l_pos.pdf(x) * prior_pos
post_neg = l_neg.pdf(x) * prior_neg

post = np.vstack([post_pos, post_neg])
evidence = post.sum(axis=0)

plt.clf()
plt.plot(x, post_pos / evidence, 'b')
plt.plot(x, post_neg / evidence, 'r')

plt.legend([r'$P(\mathcal Y=1 \mid \mathcal X)$',
            r'$P(\mathcal Y=0 \mid \mathcal X)$'])
plt.tight_layout()
plt.grid()
plt.savefig('two_classes_posteriori.png', dpi=300)


klass = lambda x: 1 if l_pos.pdf(x) * prior_pos > l_neg.pdf(x) * prior_neg else 0

plt.clf()
plt.plot([x for x, k in D if k==1 and klass(x) == 1],
         pos.pdf([x for x, k in D if k==1 and klass(x) == 1]), 'k.')
plt.plot([x for x, k in D if k==0 and klass(x) == 0],
         neg.pdf([x for x, k in D if k==0 and klass(x) == 0]), 'k.')

plt.plot([x for x, k in D if k==1 and klass(x) == 0],
         pos.pdf([x for x, k in D if k==1 and klass(x) == 0]), 'r.')
plt.plot([x for x, k in D if k==0 and klass(x) == 1],
         neg.pdf([x for x, k in D if k==0 and klass(x) == 1]), 'r.')

plt.tight_layout()
plt.grid()
plt.savefig('two_classes_posteriori_error.png', dpi=300)
