from sklearn.model_selection import train_test_split
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn import metrics
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
import seaborn as sns

credible_data = pd.read_csv("credible_data.csv")

satire_data = pd.read_csv("satire_data.csv")



total_data = pd.concat([credible_data, satire_data])

total_data = total_data[['type', 'title', 'domain']]

total_data.title.replace({r'[^\x00-\x7F]+':''}, regex=True, inplace=True)
total_data = total_data.dropna(how='any',axis=0) 

#partition randomly by type
X_train, X_test, y_train, y_test = train_test_split(total_data.drop('type',axis=1), total_data['type'], test_size=0.30, random_state=101, stratify = total_data['type'])

#partition randomly by domain
# X_train, X_test, y_train, y_test = train_test_split(total_data.drop('type',axis=1), total_data['type'], test_size=0.30, random_state=101)

# vectorizer = CountVectorizer()
vectorizer = TfidfVectorizer()


X_train = vectorizer.fit_transform(X_train.title)
X_test = vectorizer.transform(X_test.title)


logisticRegr = LogisticRegression()

logisticRegr.fit(X_train, y_train)


predictions = logisticRegr.predict(X_test)
score = logisticRegr.score(X_test, y_test)
print(score)
round_score = round(score, 5)

cm = metrics.confusion_matrix(y_test, predictions, labels=['reliable', 'satire'], normalize = 'true')
print(cm)
# sklearn.metrics.plot_confusion_matrix(logisticRegr, X_test, y_test,
#                                  display_labels=['satire', 'reliable'],
#                                  normalize='all')


cmap = sns.cubehelix_palette(50, hue=0.05, rot=0, light=0.9, dark=0, as_cmap=True)
sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square = True, cmap=cmap, cbar=False, xticklabels=['reliable', 'satire'], yticklabels=['reliable', 'satire']);
plt.ylabel('Actual label');
plt.xlabel('Predicted label');
all_sample_title = 'Accuracy Score: {0}'.format(round_score)
plt.title(all_sample_title, size = 15)

plt.show()


# pred_ytest = y_test.replace('reliable', 0)
# pred_ytest = pred_ytest.replace('satire', 1)



# probs = logisticRegr.predict_proba(X_test)
# preds = probs[:,1]

# fpr, tpr, threshold = metrics.roc_curve(pred_ytest, preds)
# roc_auc = metrics.auc(fpr, tpr)

# # method I: plt
# plt.title('Receiver Operating Characteristic')
# plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
# plt.legend(loc = 'lower right')
# plt.plot([0, 1], [0, 1],'r--')
# plt.xlim([0, 1])
# plt.ylim([0, 1])
# plt.ylabel('True Positive Rate')
# plt.xlabel('False Positive Rate')
# plt.show()

