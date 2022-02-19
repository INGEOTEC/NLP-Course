import numpy as np
from wordcloud import WordCloud as WC
from matplotlib import pylab as plt
from microtc.textmodel import TextModel
from microtc.params import OPTION_GROUP, OPTION_DELETE
from b4msa.lang_dependency import LangDependency

# %pylab inline

lang = LangDependency('spanish')
lang.filterStopWords("~la~vida~es~buena~", OPTION_GROUP)