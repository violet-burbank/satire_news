import os
import pandas as pd
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
csv_file_path = '/data/FakeNewsCorpus'
os.chdir(os.path.join(os.getcwd(), "..", "..", ".."))
os.chdir(os.path.join(os.getcwd(), "data", "FakeNewsCorpus"))
print("trying now")
data = pd.read_csv("news_cleaned_2018_02_13.csv")

print(data.head())

print("pruning")
pruned_df = data[['id', 'domain', 'type', 'url', 'title']]

print(pruned_df.head())
print("saving")

os.chdir(os.path.join(os.getcwd(), "..", ".."))
os.chdir(os.path.join(os.getcwd(), "home", "vburbank", "satire_news"))

pruned_df.to_csv('pruned_news.csv')

print("done")

# data = pd.read_csv("pruned_news.csv")

# text = data[['domain']]
# wordcloud = WordCloud(
#     width = 3000,
#     height = 2000,
#     background_color = 'black',
#     stopwords = STOPWORDS).generate(str(text))
# fig = plt.figure(
#     figsize = (40, 30),
#     facecolor = 'k',
#     edgecolor = 'k')
# plt.imshow(wordcloud, interpolation = 'bilinear')
# plt.axis('off')
# plt.tight_layout(pad=0)
# plt.show()