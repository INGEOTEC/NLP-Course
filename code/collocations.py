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
token = TokenCount(tokenizer=tm.tokenize)
for fname in glob("books/*.txt"):
    txt = open(fname, encoding="utf-8").read()
    token.process([txt])
bigrams = {k: v for k, v in token.counter.items() if k.count("~")}

cnt = Counter(bigrams)
cnt.most_common(5)
# [('of~the', 14615),
#  ('in~the', 9913),
#  ('to~the', 7339),
#  ('on~the', 4883),
#  ('and~the', 4843)]
       
# Word-cloud
wc = WC().generate_from_frequencies(bigrams)
plt.imshow(wc)
plt.axis("off")
plt.tight_layout()
plt.savefig("books_wordcloud.png", dpi=300)

# 
bigrams["in~the"]
# 9913
sum(bigrams.values())
# 2274130
N = sum(bigrams.values())
prob_bigrams = {k: v / N for k, v in bigrams.items()}
prob_bigrams["in~the"]
# 0.004359029606926605
sum(prob_bigrams.values())
# 1.0000000000142464

# Histogram
hist = defaultdict(list)
_ = [hist[v].append(k) for k, v in bigrams.items()]
plt.plot(np.log([len(hist.get(i, [0])) for i in range(1, 2000)]))
plt.grid()
plt.xlabel("Frequency")
plt.ylabel("Number of bigrams (log-scale)")
plt.tight_layout()
plt.savefig("books_hist.png", dpi=300)


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
