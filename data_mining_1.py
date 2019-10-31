import os
import pandas as pd

csv_file_path = '/data/FakeNewsCorpus'
os.chdir(os.path.join(os.getcwd(), "..", "..", ".."))
os.chdir(os.path.join(os.getcwd(), "data", "FakeNewsCorpus"))
data = pd.read_csv("news_cleaned_2018_02_13.csv")
print(data.head())

