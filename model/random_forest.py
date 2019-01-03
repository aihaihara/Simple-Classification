import json
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support, classification_report
from sklearn.externals import joblib
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

import pandas, numpy

#open data after pre-processing (JSON format)
with open("your data") as data:
    data = json.load(data)

#framing data
df = pandas.DataFrame.from_dict(data)
#sort table columns
df = df[['id', 'title', 'subtitle', 'date', 'text', 'tag', 'label']]


#convert label to numerical label
factor = pandas.factorize(df['label'])
df.label = factor[0]
definition = factor[1]

#change 'text' column to new variable (x)
x = df.iloc[:,4].values
#change 'label' column to new variable (y)
y = df.iloc[:,6].values

#splitting data into training data (60%) and testing data (40%)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.40, random_state = 21)

#uncomment script below if you want to use count vector as your feature
count_vect = CountVectorizer(analyzer='char', ngram_range=(2,3), max_features=2000)
count_vect.fit(df['text'])
x_train =  count_vect.fit_transform(x_train)
x_test =  count_vect.transform(x_test)

# #uncomment script below if you want to pickle features and using later
# #writing features file for testing new raw data
# feat = open("features.pkl", 'wb')
# pickle.dump(count_vect, feat)
# feat.close()

# #uncomment script below if you want to use tfidf as your feature
# tfidf_vect_ngram = TfidfVectorizer(analyzer='word', smooth_idf=True,ngram_range=(2,3), token_pattern=r'\w{1,}', max_features=2000)
# tfidf_vect_ngram.fit(df['text'])
# x_train =  tfidf_vect_ngram.transform(x_train)
# x_test =  tfidf_vect_ngram.transform(x_test)

# #uncomment script below if you want to pickle features and using later
# #writing features file for testing new raw data
# feat = open("features.pkl", 'wb')
# pickle.dump(tfidf_vect_ngram, feat)
# feat.close()

#train Random Forest Classifier with training data
classifier = RandomForestClassifier(n_estimators= 10, criterion='entropy', random_state=42)
classifier.fit(x_train, y_train)

#testing model with testing data 'text'
y_pred = classifier.predict(x_test)

#convert back numerical label to actual label names
reversefactor = dict(zip(range(3), definition))
y_test = numpy.vectorize(reversefactor.get)(y_test)
y_pred = numpy.vectorize(reversefactor.get)(y_pred)

#count metrics
accuracy = accuracy_score(y_test, y_pred)
score = precision_recall_fscore_support(y_test, y_pred, average='macro')
print('accuracy score:', accuracy)
print('precision, recall, and f1 score respectively:', score)
report = classification_report(y_test, y_pred)

#print confusion matrix
print(pandas.crosstab(y_test, y_pred, rownames=['Actual Label'], colnames=['Predicted Label']))
print(report)

# #uncomment if you want to write model for using it later
# model = open("randomforest.pkl", 'wb')
# pickle.dump(classifier, model)
# model.close()
print("finish!!")
