import os
import pandas as pd
import wordcloud
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt


data = pd.read_csv("dom_names.csv")

text = data[['domain']]
wordcloud = WordCloud(
    width = 300,
    height = 100,
    background_color = 'white',
    stopwords = STOPWORDS).generate(str(text))
fig = plt.figure(
    figsize = (10, 7),
    facecolor = 'k',
    edgecolor = 'k')
plt.imshow(wordcloud, interpolation = 'bilinear')
plt.axis('off')
plt.tight_layout(pad=0)
plt.show()