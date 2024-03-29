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


# total_data = pd.concat([credible_data_samp, satire_data])

# total_data = total_data[['type', 'title', 'domain']]

# total_data.title.replace({r'[^\x00-\x7F]+':''}, regex=True, inplace=True)
# total_data = total_data.dropna(how='any',axis=0) 

# #partition randomly by type
# # X_train, X_test, y_train, y_test = train_test_split(total_data.drop('type',axis=1), total_data['type'], test_size=0.30, random_state=101, stratify = total_data['type'])

# #partition randomly by domain
# X_train, X_test, y_train, y_test = train_test_split(total_data.drop('type',axis=1), total_data['type'], test_size=0.30, random_state=101, stratify = total_data['type'])

# # vectorizer = CountVectorizer()
# vectorizer = TfidfVectorizer()


# X_train = vectorizer.fit_transform(X_train.title)
# X_test = vectorizer.transform(X_test.title)


# logisticRegr = LogisticRegression()

# logisticRegr.fit(X_train, y_train)


fake_data = pd.read_csv("fake_data.csv")
conspiracy_data = pd.read_csv("conspiracy_data.csv")
clickbait_data = pd.read_csv("clickbait_data.csv")

# fake_data = fake_data.replace({r'[^\x00-\x7F]+':''})
# fake_data = fake_data.replace('', np.NaN)
# print(fake_data.title.head(10))
# fake_data = vectorizer.transform(fake_data.title.values.astype('str'))


# fake_preds = logisticRegr.predict(fake_data)
# print(fake_preds)

# fake_reliable_count = np.count_nonzero(fake_preds == 'reliable')
# fake_satire_count = np.count_nonzero(fake_preds == 'satire')

# fake_pred_counts = [fake_reliable_count, fake_satire_count]
# labels = ['reliable', 'satire']
# # plt.pie(fake_pred_counts, labels=labels, startangle=90, autopct='%.1f%%')
# # plt.title('Fake News Predictions')
# # plt.show()

# conspiracy_data = conspiracy_data.replace({r'[^\x00-\x7F]+':''})
# conspiracy_data = conspiracy_data.replace('', np.NaN)
# print(conspiracy_data.title.head(10))
# conspiracy_data = vectorizer.transform(conspiracy_data.title.values.astype('str'))


# conspiracy_preds = logisticRegr.predict(conspiracy_data)
# print(conspiracy_data)

# conspiracy_reliable_data = np.count_nonzero(conspiracy_preds == 'reliable')
# conspiracy_satire_count = np.count_nonzero(conspiracy_preds == 'satire')

# conspiracy_pred_counts = [conspiracy_reliable_data, conspiracy_satire_count]
# # plt.pie(conspiracy_pred_counts, labels=labels, startangle=90, autopct='%.1f%%')
# # plt.title('Conspiracy News Predictions')
# # plt.show()


# clickbait_data = clickbait_data.replace({r'[^\x00-\x7F]+':''})
# clickbait_data = clickbait_data.replace('', np.NaN)
# print(clickbait_data.title.head(10))
# clickbait_data = vectorizer.transform(clickbait_data.title.values.astype('str'))


# clickbait_preds = logisticRegr.predict(clickbait_data)
# print(clickbait_data)

# clickbait_reliable_data = np.count_nonzero(clickbait_preds == 'reliable')
# clickbait_satire_count = np.count_nonzero(clickbait_preds == 'satire')

# clickbait_pred_counts = [clickbait_reliable_data, clickbait_satire_count]
# # plt.pie(clickbait_pred_counts, labels=labels, startangle=90, autopct='%.1f%%')
# # plt.title('Clickbait News Predictions')
# # plt.show()
# barWidth = 0.25
 
# # set height of bar
# bars1 = [conspiracy_satire_count, fake_satire_count, clickbait_satire_count]
# bars2 = [conspiracy_reliable_data, fake_reliable_count, clickbait_reliable_data]
 
# # Set position of bar on X axis
# r1 = np.arange(len(bars1))
# r2 = [x + barWidth for x in r1]
 
# # Make the plot
# plt.bar(r1, bars1, color='#FF8000', width=barWidth, edgecolor='white', label='Satire')
# plt.bar(r2, bars2, color='#000080', width=barWidth, edgecolor='white', label='Reliable')
 
# # Add xticks on the middle of the group bars
# plt.xlabel('Type', fontweight='bold')
# plt.ylabel('Number of Classifications', fontweight = 'bold')
# plt.title('Classification as Reliable or Satire')
# plt.xticks([r + barWidth for r in range(len(bars1))], ['Conspiracy', 'Fake News', 'Clickbait'])
 
# # Create legend & Show graphic
# plt.legend()
# plt.show()

# predictions = logisticRegr.predict(X_test)
# score = logisticRegr.score(X_test, y_test)
# round_score = round(score, 5)

# print(score)


# cm = metrics.confusion_matrix(y_test, predictions, labels=['reliable', 'satire'], normalize = 'true')


# plt.figure(figsize=(5,5))
# cmap = sns.cubehelix_palette(50, hue=0.05, rot=0, light=0.9, dark=0, as_cmap=True)
# sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square = True, cmap=cmap, cbar=False, xticklabels=['reliable', 'satire'], yticklabels=['reliable', 'satire']);
# plt.ylabel('Actual label');
# plt.xlabel('Predicted label');
# all_sample_title = 'Accuracy Score: {0}'.format(round_score)
# plt.title(all_sample_title, size = 15);
# plt.show()

# credible_data.title.replace({r'[^\x00-\x7F]+':''}, regex=True, inplace=True)
# credible_data = credible_data.dropna(how='any',axis=0) 

# y_test = credible_data.type
# total_cred = vectorizer.transform(credible_data.title)

# print(len(y_test))

# score = logisticRegr.score(total_cred, y_test)
# print(score)

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


total_data = pd.concat([credible_data_samp, satire_data, conspiracy_data, fake_data, clickbait_data])

total_data = total_data[['type', 'title', 'domain']]

total_data.title.replace({r'[^\x00-\x7F]+':''}, regex=True, inplace=True)
total_data = total_data.dropna(how='any',axis=0) 

#partition randomly by type
# X_train, X_test, y_train, y_test = train_test_split(total_data.drop('type',axis=1), total_data['type'], test_size=0.30, random_state=101, stratify = total_data['type'])

#partition randomly by domain
X_train, X_test, y_train, y_test = train_test_split(total_data.drop('type',axis=1), total_data['type'], test_size=0.05, random_state=101, stratify = total_data['type'])

# vectorizer = CountVectorizer()
vectorizer = TfidfVectorizer()


X_train = vectorizer.fit_transform(X_train.title)
X_test = vectorizer.transform(X_test.title)

logisticRegr = LogisticRegression()

logisticRegr.fit(X_train, y_train)

classes = logisticRegr.classes_
class_list = list(map(lambda el:[el], classes)) 
class_list_np = np.asarray(class_list)

coeffs = logisticRegr.coef_
print(type(coeffs))
print(coeffs.shape)

model_features = list(vectorizer.vocabulary_.keys())
# model_features_list = list(map(lambda el:[el], model_features)) 
# model_features_np = np.asarray(model_features_list)
# print(len(model_features_np))
# print(model_features_np)

coeffs_feat = np.vstack((model_features, coeffs))

coeffs_feat_class = np.append(class_list_np, coeffs_feat, axis=1)

print(coeffs_feat_class)

# coef_dict_0 = {}
# for coef, feat in zip(coeffs[0,:],model_features):
#     coef_dict_0[feat] = coef

# keys = list(coef_dict_0.keys())[:10]

# print(dict((k, coef_dict_0[k]) for k in (keys)))

# top_coef_0 = dict(sorted(coef_dict_0, key=coef_dict_0.get, reverse=False)[:6])
# top_coef_0 = top_coef_0.keys()

# coef_dict_1 = {}
# for coef, feat in zip(coeffs[1,:],model_features):
#     coef_dict_1[feat] = coef

# top_coef_1 = dict(sorted(coef_dict_1, key=coef_dict_1.get, reverse=False)[:6])
# top_coef_1 = top_coef_1.keys()

# coef_dict_2 = {}
# for coef, feat in zip(coeffs[2,:],model_features):
#     coef_dict_2[feat] = coef

# top_coef_2 = dict(sorted(coef_dict_2, key=coef_dict_2.get, reverse=False)[:6])
# top_coef_2 = top_coef_2.keys()

# coef_dict_3 = {}
# for coef, feat in zip(coeffs[3,:],model_features):
#     coef_dict_3[feat] = coef

# top_coef_3 = dict(sorted(coef_dict_3, key=coef_dict_3.get, reverse=False)[:6])
# top_coef_3 = top_coef_3.keys()

# coef_dict_4 = {}
# for coef, feat in zip(coeffs[4,:],model_features):
#     coef_dict_4[feat] = coef

# top_coef_4 = dict(sorted(coef_dict_4, key=coef_dict_4.get, reverse=False)[:6])
# top_coef_4 = top_coef_4.keys()

# weight_map = np.zeros((5,25))
# for i in range(0, 6):
#     weight_map[0][0 + (5 * i)] = coeff_dict_0[top_coef_0[i]]
#     weight_map[0][1 + (5 * i)] = coeff_dict_0[top_coef_1[i]]
#     weight_map[0][2 + (5 * i)] = coeff_dict_0[top_coef_2[i]]
#     weight_map[0][3 + (5 * i)] = coeff_dict_0[top_coef_3[i]]
#     weight_map[0][4 + (5 * i)] = coeff_dict_0[top_coef_4[i]]

#     weight_map[1][0 + (5 * i)] = coeff_dict_1[top_coef_0[i]]
#     weight_map[1][1 + (5 * i)] = coeff_dict_1[top_coef_1[i]]
#     weight_map[1][2 + (5 * i)] = coeff_dict_1[top_coef_2[i]]
#     weight_map[1][3 + (5 * i)] = coeff_dict_1[top_coef_3[i]]
#     weight_map[1][4 + (5 * i)] = coeff_dict_1[top_coef_4[i]]

#     weight_map[2][0 + (5 * i)] = coeff_dict_2[top_coef_0[i]]
#     weight_map[2][1 + (5 * i)] = coeff_dict_2[top_coef_1[i]]
#     weight_map[2][2 + (5 * i)] = coeff_dict_2[top_coef_2[i]]
#     weight_map[2][3 + (5 * i)] = coeff_dict_2[top_coef_3[i]]
#     weight_map[2][4 + (5 * i)] = coeff_dict_2[top_coef_4[i]]

#     weight_map[3][0 + (5 * i)] = coeff_dict_3[top_coef_0[i]]
#     weight_map[3][1 + (5 * i)] = coeff_dict_3[top_coef_1[i]]
#     weight_map[3][2 + (5 * i)] = coeff_dict_3[top_coef_2[i]]
#     weight_map[3][3 + (5 * i)] = coeff_dict_3[top_coef_3[i]]
#     weight_map[3][4 + (5 * i)] = coeff_dict_3[top_coef_4[i]]

#     weight_map[4][0 + (5 * i)] = coeff_dict_4[top_coef_0[i]]
#     weight_map[4][1 + (5 * i)] = coeff_dict_4[top_coef_1[i]]
#     weight_map[4][2 + (5 * i)] = coeff_dict_4[top_coef_2[i]]
#     weight_map[4][3 + (5 * i)] = coeff_dict_4[top_coef_3[i]]
#     weight_map[4][4 + (5 * i)] = coeff_dict_4[top_coef_4[i]]

# print(weight_map)

# result = coeffs[:, np.flip(np.argsort(coeffs.sum(axis=0)))]

	
# words = result[ : , 0: 30]

# print(words)

# plt.imshow(words, cmap='hot', interpolation='nearest')
# plt.show()