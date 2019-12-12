import matplotlib.pyplot as plt
import pandas as pd
import os
import wordcloud
from wordcloud import WordCloud, STOPWORDS

print("plotting")
# data = pd.read_csv('pruned_news.csv')

data = pd.read_csv('pruned_news.csv')

# data = data[data.type.str.contains("2018") == False]
# data = data[data.type.str.contains("2019") == False] 
# data = data[data.type.str.contains("2017") == False]
# data = data[data.type.str.contains("Iraq") == False]
# data = data[data.type.str.contains("Linton") == False]


# data = data[data.type != 'unreliable']
# data = data[data.type != 'bias']
# data = data[data.type != 'rumor']

# data[['domain']] = [ x.split('.') for x in data['domain']]
# stop = ["com", "www", "org", "co", "uk", "aus", "domain", "columns", "rows", "au", "af", "ca", "go", "de", "in", "nz", "m", "net"]

# data[['domain']] = data['domain'].apply(lambda x: [item for item in x if item not in stop])

# data[['domain']] = [" ".join(x) for x in data['domain']]

# fig, ax = plt.subplots()
# data['type'].value_counts().plot(ax=ax, kind='bar')
# print(data['type'].value_counts())
# print("plotted")
# plt.show()

# Generate a word cloud image

# wordcloud = WordCloud(stopwords=["com", "www", "org", "co", "uk", "aus", "domain", "columns", "rows"], background_color="white").generate(str(text))

# # Display the generated image:
# # the matplotlib way:
# plt.imshow(wordcloud, interpolation='bilinear')
# plt.axis("off")
# plt.show()

# fig, ax = plt.subplots()
# data['type'].value_counts().plot(ax=ax, kind='bar')
# print(data['type'].value_counts())
# print("plotted")
# plt.show()

# credible_data = data[data.type == 'reliable']
# print(credible_data.head(10))

# satire_data = data[data.type == 'satire']
# print(satire_data.head(10))

# wordcloud = WordCloud(stopwords=["com", "www", "org", "co", "uk", "aus", "domain", "columns", "rows", "au", "af", "ca", "go", "de", "in"], background_color="white").generate(str(credible_data[['domain']]))

# # Display the generated image:
# # the matplotlib way:
# plt.imshow(wordcloud, interpolation='bilinear')
# plt.axis("off")
# plt.show()

# fig, ax = plt.subplots()
# credible_data['domain'].value_counts().plot(ax=ax, kind='bar')
# print(credible_data['domain'].value_counts())
# print("plotted")
# plt.show()

# wordcloud = WordCloud(stopwords=["com", "www", "org", "co", "uk", "aus", "domain", "columns", "rows", "au", "af", "ca", "go", "de", "in"], background_color="white").generate(str(satire_data[['domain']]))

# # Display the generated image:
# # the matplotlib way:
# plt.imshow(wordcloud, interpolation='bilinear')
# plt.axis("off")
# plt.show()

# fig, ax = plt.subplots()
# satire_data['domain'].value_counts().plot(ax=ax, kind='bar')
# print(satire_data['domain'].value_counts())
# print("plotted")
# plt.show()

#eliminate non-unicode characters
# credible_data.title.replace({r'[^\x00-\x7F]+':''}, regex=True, inplace=True)
# satire_data.title.replace({r'[^\x00-\x7F]+':''}, regex=True, inplace=True)

# credible_data.to_csv('credible_data.csv')
# satire_data.to_csv('satire_data.csv')

fake_data = data[data.type == 'fake']
print(fake_data.head(10))

conspiracy_data = data[data.type == 'conspiracy']
print(conspiracy_data.head(10))

clickbait_data = data[data.type == 'clickbait']
print(clickbait_data.head(10))

fake_data.title.replace({r'[^\x00-\x7F]+':''}, regex=True, inplace=True)
conspiracy_data.title.replace({r'[^\x00-\x7F]+':''}, regex=True, inplace=True)
clickbait_data.title.replace({r'[^\x00-\x7F]+':''}, regex=True, inplace=True)

fake_data.to_csv('fake_data.csv')
conspiracy_data.to_csv('conspiracy_data.csv')
clickbait_data.to_csv('clickbait_data.csv')

