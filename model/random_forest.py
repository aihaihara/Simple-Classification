import json
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support, classification_report
from sklearn.externals import joblib
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

import pandas, numpy

with open("test2.json") as data:
    data = json.load(data)

df = pandas.DataFrame.from_dict(data)
df = df[['id', 'title', 'subtitle', 'date', 'text', 'tag', 'label']]
#print(df)
factor = pandas.factorize(df['label'])
#print(factor)
df.label = factor[0]
definition = factor[1]
# print(df.label.head())
# print(definition)
x = df.iloc[:,4].values
y = df.iloc[:,6].values
#print(x)
#print(y)
# print("independent:")
# print(x[:5])
# print("dependent:")
# print(y[:7])

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.40, random_state = 21)
#print(x_train)
# count_vect = CountVectorizer(analyzer='char', ngram_range=(2,3), max_features=2000)
# count_vect.fit(df['text'])
# #print(count_vect)
# x_train =  count_vect.fit_transform(x_train)
# x_test =  count_vect.transform(x_test)

# ngram level tf-idf 
tfidf_vect_ngram = TfidfVectorizer(analyzer='word', smooth_idf=True,ngram_range=(2,3), token_pattern=r'\w{1,}', max_features=2000)
tfidf_vect_ngram.fit(df['text'])
x_train =  tfidf_vect_ngram.transform(x_train)
x_test =  tfidf_vect_ngram.transform(x_test)

classifier = RandomForestClassifier(n_estimators= 10, criterion='entropy', random_state=42)
classifier.fit(x_train, y_train)

y_pred = classifier.predict(x_test)
reversefactor = dict(zip(range(3), definition))
y_test = numpy.vectorize(reversefactor.get)(y_test)
y_pred = numpy.vectorize(reversefactor.get)(y_pred)
# print(y_pred)
# print(y_test)
accuracy = accuracy_score(y_test, y_pred)
score = precision_recall_fscore_support(y_test, y_pred, average='macro')
print('accuracy score:', accuracy)
print('precision, recall, and f1 score respectively:', score)
report = classification_report(y_test, y_pred)
#print(list(zip(df.columns[0:6], classifier.feature_importances_)))
print(pandas.crosstab(y_test, y_pred, rownames=['Actual Label'], colnames=['Predicted Label']))
print(report)
#writing model
model = open("randomforest.pkl", 'wb')
pickle.dump(classifier, model)
model.close()
print("finish!!")
