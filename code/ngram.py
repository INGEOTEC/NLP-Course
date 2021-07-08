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
from collections import Counter
from typing import Iterable


class Read(object):
    def __init__(self, fnames: list, n_gram: int=2) -> None:
        np.random.seed(0)
        self._fnames = fnames
        self.test_set = []
        self.n_gram = n_gram

    def read(self) -> Iterable:
        def process(lst: list) -> str:
            L = [lst.pop() for _ in range(len(lst))]
            L.reverse()
            line = "".join(L)
            if self.n_gram <= 1:
                return line
            frst = " ".join(["#p"] * (self.n_gram - 1))
            scnd = " ".join(["p#"] * (self.n_gram - 1))
            return "%s %s %s" % (frst, line, scnd)

        for fname in self._fnames:
            L = []
            for line in open(fname).readlines():
                if line.count("*** END OF THE PROJECT GUTENBERG"):
                    if len(L):
                        if np.random.uniform() < 0.10:
                            self.test_set.append(process(L))
                        else:
                            yield process(L)
                    break
                if line == "\n" and len(L):
                    if np.random.uniform() < 0.10:
                        self.test_set.append(process(L))
                        continue
                    yield process(L)
                elif len(line):
                    _ = line.strip()
                    if len(_):
                        L.append(_)
            if len(L):
                if np.random.uniform() < 0.10:
                    self.test_set.append(process(L))
                else:
                    yield process(L)


class LM(object):
    def __init__(self, data, words: bool=False) -> None:
        self._words = words
        self._data = data
        if words:
            self.N = sum(data.values())
        else:
            self.__init()

    def __init(self) -> None:
        N = Counter()
        for k, v in self._data.items():
            words = "~".join(k.split("~")[:-1])
            N.update({words: v})
        self.N = N

    def log_prob(self, ngram: str) -> float:
        c1 = self._data[ngram]
        if self._words:
            c2 = self.N
        else:
            words = "~".join(ngram.split("~")[:-1])
            c2 = self.N[words]
        if c1 and c2:
            return np.log(c1) - np.log(c2)
        raise ValueError("ngram %s not found" % ngram) 

    def prob(self, ngram: str) -> float:
        return np.exp(self.log_prob(ngram))


tm = TokenCount.textModel(token_list=[-1])
token = TokenCount(tokenizer=tm.tokenize)
read = Read(glob("books/*.txt"),
            n_gram=tm.token_list[0] * -1)
token.process(read.read())

lm = LM(token.counter, words=tm.token_list[0] == -1)

logp = 0
max_logp, cnt = 0, 0
N = 0
for txt in read.test_set:
    for ngram in tm.tokenize(txt):
        N += 1
        try:
            _ = lm.log_prob(ngram)
            if _ < max_logp:
                max_logp = _
            logp -= _
        except ValueError:
            cnt += 1
logp -= max_logp * cnt
pp = np.exp(logp / N)
pp