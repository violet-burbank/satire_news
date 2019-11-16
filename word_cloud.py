import os
import pandas as pd
import numpy as np

from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt


credible_data = pd.read_csv("credible_data.csv")

satire_data = pd.read_csv("satire_data.csv")


credible_freqs = credible_data.groupby(["domain"]).count()
credible_freqs.rename(columns={'Unnamed: 0':'count'}, inplace=True)
credible_freqs['log_freq'] = np.log(credible_freqs['count'])
credible_freqs = credible_freqs.drop(["id", "type", "url", "count", "title"], axis=1)
credible_freqs = credible_freqs.round(0)
credible_freqs = credible_freqs[credible_freqs.log_freq != 0]
print(credible_freqs)
mydict = credible_freqs.to_dict()['log_freq']


wordcloud = WordCloud()
wordcloud.generate_from_frequencies(mydict)
plt.figure()
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()

satire_freqs = satire_data.groupby(["domain"]).count()
satire_freqs.rename(columns={'Unnamed: 0':'count'}, inplace=True)
satire_freqs['log_freq'] = np.log(satire_freqs['count'])
satire_freqs = satire_freqs.drop(["id", "type", "url", "count", "title"], axis=1)
satire_freqs = satire_freqs.round(0)
satire_freqs = satire_freqs[satire_freqs.log_freq != 0]
mydict = satire_freqs.to_dict()['log_freq']

wordcloud = WordCloud()
wordcloud.generate_from_frequencies(mydict)
plt.figure()
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()

