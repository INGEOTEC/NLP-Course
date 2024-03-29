# Copyright 2021 Mario Graff Guerrero

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import numpy as np
from text_models.dataset import TokenCount
from glob import glob
from text_models.utils import TStatistic, LikelihoodRatios
from wordcloud import WordCloud as WC 
from matplotlib import pylab as plt
from collections import defaultdict, Counter
from text_models import Vocabulary

tm = TokenCount.textModel(token_list=[-2, -1])
tm.tokenize("Good afternoon, we will make some nice word clouds.")
## SHOW - bigrams "w_{i}~w_{i+1}"

token = TokenCount(tokenizer=tm.tokenize)
for fname in glob("books/*.txt"):
    txt = open(fname, encoding="utf-8").read()
    token.process([txt])
token.counter
## SHOW - number of times each bigram and word appear

bigrams = {k: v for k, v in token.counter.items() if k.count("~")}
cnt = Counter(bigrams)
cnt.most_common(5)
# [('of~the', 14615),
#  ('in~the', 9913),
#  ('to~the', 7339),
#  ('on~the', 4883),
#  ('and~the', 4843)]
bigrams
## SHOW the selection
       
# Word-cloud
wc = WC().generate_from_frequencies(bigrams)
plt.imshow(wc)
plt.axis("off")
plt.tight_layout()
## SHOW
plt.savefig("books_wordcloud.png", dpi=300)

## SKIP
# Histogram
hist = defaultdict(list)
_ = [hist[v].append(k) for k, v in bigrams.items()]
plt.plot(np.log([len(hist.get(i, [0])) for i in range(1, 2000)]))
plt.grid()
plt.xlabel("Frequency")
plt.ylabel("Number of bigrams (log-scale)")
plt.tight_layout()
plt.savefig("books_hist.png", dpi=300)
## END OF SKIP

## Wald test
bigrams["in~the"]
# 12670
sum([x for x in bigrams.values()])
# 2982163
# Total number of bigrams
## SHOW
N = sum([x for x in bigrams.values()])
prob_bigrams = {k: v / N for k, v in bigrams.items()}
prob_bigrams["in~the"]
# 0.004248594057400619
## SHOW
sum([x for x in prob_bigrams.values()])
# 0.9999999999999999

words = {k: v for k, v in token.counter.items() if k.count("~") == 0}
wald = dict()
for w, v in prob_bigrams.items():
    w1, w2 = w.split("~")
    se = np.sqrt(v * (1 - v) / N)
    p1 = words.get(w1, 1) / N
    p2 = words.get(w2, 1) / N
    ##
    # Wald Test
    ##
    wald[w] = (v - (p1 * p2)) / se

wc = WC().generate_from_frequencies(wald)
plt.imshow(wc)
plt.axis("off")
plt.tight_layout()
## SHOW
plt.savefig("books_wald.png", dpi=300)

## Frequency vs Wald
for key, value in cnt.most_common(5):
    print(key, value, wald[key])
## SHOW

keys = list(wald.keys())
plt.plot([bigrams[x] for x in keys], [wald[x] for x in keys], '.')
plt.grid()
plt.xlabel("Frequency")
plt.xlabel("Wald Test without the absolute")
## SHOW

keys = list(wald.keys())
plt.plot([bigrams[x] for x in keys], [np.fabs(wald[x]) for x in keys], '.')
plt.grid()
plt.xlabel("Frequency")
plt.xlabel("Wald Test")
## SHOW

keys = [k for k, v in wald.items() if v > 0]
plt.plot([bigrams[x] for x in keys], [wald[x] for x in keys], '.')
plt.grid()
plt.xlabel("Frequency")
plt.ylabel("Wald Test (>0)")
## SHOW

## SKIP
wald_max = dict()
for w, v in prob_bigrams.items():
    w1, w2 = w.split("~")
    se = np.sqrt(v * (1 - v) / N)
    p1 = words.get(w1, 1) / N
    p2 = words.get(w2, 1) / N
    w_value = (v - (p1 * p2)) / se
    freq = bigrams[w]
    if wald_max.get(freq, 0) < w_value:
        wald_max[freq] = w_value

x = list(wald_max.keys())
x.sort()
plt.plot([wald_max[k] for k in x if k <=100], '-')
plt.xlabel("Frequency")
plt.ylabel("Max Wald statistic")
plt.savefig("max_wald.png", dpi=300)
## END OF SKIP

_ = {k: -v for k, v in wald.items()}
wc = WC().generate_from_frequencies(_)
plt.imshow(wc)
plt.axis("off")
plt.tight_layout()


_ = {k: -v for k, v in wald.items() if bigrams[k] == 15}
wc = WC().generate_from_frequencies(_)
plt.imshow(wc)
plt.axis("off")
plt.tight_layout()
## SHOW
plt.savefig("max_wald_15.png", dpi=300)

txt = "***** This file should be named 345-0.txt or 345-0.zip *****"
tm.tokenize(txt)
## SHOW

_ = {k: v for k, v in wald.items() if bigrams[k] == 25}
wc = WC().generate_from_frequencies(_)
plt.imshow(wc)
plt.axis("off")
plt.tight_layout()
## SHOW

## negative
_ = {k: -v for k, v in wald.items() if bigrams[k] == 15}
wc = WC().generate_from_frequencies(_)
plt.imshow(wc)
plt.axis("off")
plt.tight_layout()
## SHOW

## GO to grep

# Collocations
tstats = TStatistic(token.counter)

_ = [max([tstats.compute(k) for k in hist[i]]) for i in range(1, 100)]
plt.plot(_)
plt.grid()
plt.xlabel("Frequency")
plt.ylabel("Max t-statistic")
plt.tight_layout()
plt.savefig("books_t_stats.png", dpi=300)

# Collocations Likelihood ratios
tstats2 = LikelihoodRatios(token.counter)

fig, axs = plt.subplots(2)
for i, ins in enumerate([tstats, tstats2]):
    _ = {k: ins.compute(k) for k in bigrams}
    wc = WC().generate_from_frequencies(_)
    axs[i].imshow(wc)
    axs[i].axis("off")
plt.tight_layout()    
plt.savefig("wordcloud_comparison.png", dpi=300)

fig, axs = plt.subplots(2)
for i, ins in enumerate([tstats, tstats2]):
    _ = {k: ins.compute(k) for k in hist[5]}
    wc = WC().generate_from_frequencies(_)
    axs[i].imshow(wc)
    axs[i].axis("off")
plt.tight_layout()    
plt.savefig("wordcloud_comparison_15.png", dpi=300)

_ = [max([tstats2.compute(k) for k in hist[i]]) for i in range(1, 100)]
plt.plot(_)
plt.grid()
plt.xlabel("Frequency")
plt.ylabel("Max t-statistic")
plt.tight_layout()

_ = {k: tstats.compute(k) for k in bigrams}
wc = WC().generate_from_frequencies(_)
plt.imshow(wc)
plt.axis("off")
plt.tight_layout()


# Word-cloud
_ = {k: tstats.compute(k) for k in hist[15]}
wc = WC().generate_from_frequencies(_)
plt.imshow(wc)
plt.tight_layout()
plt.axis("off")
plt.tight_layout()
plt.savefig("books_wordcloud_15.png", dpi=300)


# All words
D = dict()
for k in range(10, 50):
    words = hist[k]
    stats = [tstats.compute(w) for w in words]
    if len(stats) < 10:
        continue
    den = np.median(stats)
    mean = np.mean(stats)
    den = den if den > mean else mean
    for w, v in zip(words, stats):
        if v < 2.576:
            continue
        D[w] = v / den
wc = WC().generate_from_frequencies(D)
plt.imshow(wc)
plt.axis("off")
plt.tight_layout()
plt.savefig("books_wordcloud_all.png", dpi=300)


# Using another corpus

# token = TokenCount(tokenizer=tm.tokenize)
# fname = glob("books/*.txt")[0]
# txt = open(fname, encoding="utf-8").read()
# token.process([txt]) 

books = Vocabulary(token.counter)
books.probability()
baseline = Vocabulary([dict(year=2021, month=2, day=day) for day in [3, 5, 20]], lang="En", country="US")
baseline.probability()

_ = {k: v / baseline.voc.get(k, 1e-5) for k, v in books.items() if k.count("~")}
wc = WC().generate_from_frequencies(_)
plt.imshow(wc)
plt.axis("off")
plt.tight_layout()
plt.savefig("books_wordcloud_allv2.png", dpi=300)

# Twitter
voc = Vocabulary(dict(year=2021, month=2, day=14), lang="En", country="US")
bigrams = {k: v for k, v in voc.voc.items() if k.count("~")}

# Histogram
hist = defaultdict(list)
[hist[v].append(k) for k, v in bigrams.items()]
plt.plot(np.log([len(hist.get(i, [0])) for i in range(1, 2000)]))

tstats = TStatistic(voc.voc)
_ = [max([tstats.compute(k) for k in hist[i]]) for i in range(1, 300) if len(hist[i]) > 10]
plt.plot(_)
plt.grid()
plt.xlabel("Frequency")
plt.ylabel("Max t-statistic")
plt.tight_layout()
plt.savefig("tweets_t_stats.png", dpi=300)

# 
D = dict()
for k in range(10, 50):
    words = hist[k]
    stats = [tstats.compute(w) for w in words]
    den = np.median(stats)
    mean = np.mean(stats)
    den = den if den > mean else mean    
    for w, v in zip(words, stats):
        if v < 2.576:
            continue
        D[w] = v / den
wc = WC().generate_from_frequencies(D)
plt.imshow(wc)
plt.axis("off")
plt.tight_layout()
plt.savefig("tweets_wordcloud_all.png", dpi=300)

voc.probability()
wc = WC().generate_from_frequencies({k: v / baseline.voc.get(k, 1e-5) for k, v in voc.items() if k.count("~")})
plt.imshow(wc)
plt.axis("off")
plt.tight_layout()
plt.savefig("tweets_wordcloud_allv2.png", dpi=300)

prev = Vocabulary(dict(year=2020, month=2, day=14), lang="En", country="US")
prev.probability()

wc = WC().generate_from_frequencies({k: v / prev.voc.get(k, 1e-5) for k, v in voc.items() if k.count("~")})
plt.imshow(wc)
plt.axis("off")
plt.tight_layout()
plt.savefig("tweets_wordcloud_prev.png", dpi=300)

# Comparison
tstats2 = LikelihoodRatios(voc.voc)
tstats = TStatistic(voc.voc)

fig, axs = plt.subplots(2)
for i, ins in enumerate([tstats, tstats2]):
    _ = {k: ins.compute(k) for k in voc if k.count("~")}
    wc = WC().generate_from_frequencies(_)
    axs[i].imshow(wc)
    axs[i].axis("off")
plt.tight_layout()    
plt.savefig("wordcloud_comparison_tweets.png", dpi=300)


_ = {k: v for k, v in voc.items() if k.count("~")}
wc = WC().generate_from_frequencies(_)
plt.imshow(wc)
plt.axis("off")
plt.tight_layout()   
plt.savefig("wordcloud_tweets.png", dpi=300)