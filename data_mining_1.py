import os
import pandas as pd

csv_file_path = '/data/FakeNewsCorpus'
os.chdir(os.path.join(os.getcwd(), "..", "..", ".."))
os.chdir(os.path.join(os.getcwd(), "data", "FakeNewsCorpus"))
print("trying now")
data = pd.read_csv("news_cleaned_2018_02_13.csv")
print(data.head())

pruned_df = data['id', 'domain', 'type', 'url', 'title']

print(pruned_df.head())

pruned_df.to_csv('pruned_news.csv')

print("done")
