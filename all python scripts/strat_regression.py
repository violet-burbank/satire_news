from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn import metrics
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
import seaborn as sns

credible_data = pd.read_csv("credible_data.csv")

satire_data = pd.read_csv("satire_data.csv")

credible_data_samp = credible_data.groupby('domain').apply(lambda x: x.sample(frac=0.06))


total_data = pd.concat([credible_data_samp, satire_data])

total_data = total_data[['type', 'title', 'domain']]

total_data.title.replace({r'[^\x00-\x7F]+':''}, regex=True, inplace=True)
total_data = total_data.dropna(how='any',axis=0) 

#partition randomly by type
# X_train, X_test, y_train, y_test = train_test_split(total_data.drop('type',axis=1), total_data['type'], test_size=0.30, random_state=101, stratify = total_data['type'])

#partition randomly by domain
X_train, X_test, y_train, y_test = train_test_split(total_data.drop('type',axis=1), total_data['type'], test_size=0.30, random_state=101, stratify = total_data['domain'])

# vectorizer = CountVectorizer()
vectorizer = TfidfVectorizer()


X_train = vectorizer.fit_transform(X_train.title)
X_test = vectorizer.transform(X_test.title)


logisticRegr = LogisticRegression()

logisticRegr.fit(X_train, y_train)
predictions = logisticRegr.predict(X_test)
score = logisticRegr.score(X_test, y_test)

print(score)


cm = metrics.confusion_matrix(y_test, predictions)
print(cm)

# print(cm[0][1].head())
# print(cm[1][0].head())

plt.figure(figsize=(5,5))
sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'Blues_r');
plt.ylabel('Actual label');
plt.xlabel('Predicted label');
all_sample_title = 'Accuracy Score: {0}'.format(score)
plt.title(all_sample_title, size = 15);
plt.show()

# credible_data.title.replace({r'[^\x00-\x7F]+':''}, regex=True, inplace=True)
# credible_data = credible_data.dropna(how='any',axis=0) 

# y_test = credible_data.type
# total_cred = vectorizer.transform(credible_data.title)

# print(len(y_test))

# score = logisticRegr.score(total_cred, y_test)
# print(score)

pred_ytest = y_test.replace('reliable', 0)
pred_ytest = pred_ytest.replace('satire', 1)



probs = logisticRegr.predict_proba(X_test)
preds = probs[:,1]

fpr, tpr, threshold = metrics.roc_curve(pred_ytest, preds)
roc_auc = metrics.auc(fpr, tpr)

# method I: plt
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()