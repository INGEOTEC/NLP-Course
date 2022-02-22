import numpy as np
from wordcloud import WordCloud as WC
from matplotlib import pylab as plt
from microtc.textmodel import TextModel
from microtc.params import OPTION_GROUP, OPTION_DELETE
from b4msa.lang_dependency import LangDependency
import re

# %pylab inline

lang = LangDependency('spanish')
lang.filterStopWords("~la~vida~es~buena~", OPTION_GROUP)

text = 'Hi @xx, @mm is talking about you.'
re.sub(r"@\S+", "", text)

text = 'Hi @xx, @mm is talking about you.'
re.sub(r"@\S+", "_usr", text)

text = "go http://google.com, and find out"
re.sub(r"https?://\S+", "", text)

text = "we have won 10 M"
re.sub(r"\d+\.?\d+", "_num", text)
