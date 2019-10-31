# ssh vburbank@134.173.191.241 -p 5055
import os
import pandas as pd

data = pd.read_csv("pruned_news.csv")

print("pruning")
pruned_df = data[data.type != 'unreliable']
pruned_df = pruned_df[pruned_df.type != 'bias']  

domain_names = pruned_df[['domain']]

domain_names.to_csv('dom_names.csv')