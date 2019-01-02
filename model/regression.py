import json
import pickle
import pandas, numpy
from sklearn import metrics
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support, classification_report
from sklearn.externals import joblib
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer


#open data after pre-processing (JSON format)
with open("your data") as data:
data = json.load(data)

#framing data
df = pandas.DataFrame.from_dict(data)

#convert label to numerical label
factor = pandas.factorize(df['label'])
df.label = factor[0]
definition = factor[1]

#splitting data into training data (60%) and testing data (40%)
train_x, test_x, train_y, test_y = train_test_split(df['text'], df['label'], test_size = 0.40, random_state = 21)

# #uncomment script below if you want to use count vector as your feature
# count_vect = CountVectorizer(analyzer='char', ngram_range=(2,3), max_features=2000)
# count_vect.fit(df['text'])
# #print(count_vect)
# x_train =  count_vect.fit_transform(x_train)
# x_test =  count_vect.transform(x_test)

#uncomment script below if you want to use tfidf as your feature
tfidf_vect_ngram = TfidfVectorizer(analyzer='char', ngram_range=(2,3), token_pattern=r'\w{1,}', max_features=2000)
tfidf_vect_ngram.fit(df['text'])
train_x =  tfidf_vect_ngram.transform(train_x)
test_x =  tfidf_vect_ngram.transform(test_x)

#train Logistic Regression Classifier with training data
classifier = linear_model.LogisticRegression()
classifier.fit(train_x, train_y)

#testing model with testing data 'text'
pred_y = classifier.predict(test_x)

#convert back numerical label to actual label names
reversefactor = dict(zip(range(3), definition))
test_y = numpy.vectorize(reversefactor.get)(test_y)
pred_y = numpy.vectorize(reversefactor.get)(pred_y)

#count metrics
accuracy = accuracy_score(test_y, pred_y)
score = precision_recall_fscore_support(test_y, pred_y, average='macro')
print('accuracy score:', accuracy)
print('precision, recall, and f1 score respectively:', score)
report = classification_report(test_y, pred_y)

#print confusion matrix
print(pandas.crosstab(test_y, pred_y, rownames=['Actual Label'], colnames=['Predicted Label']))
print(report)

# #uncomment if you want to write model for using it later
# model = open("regression.pkl", 'wb')
# pickle.dump(classifier, model)
# model.close()
print("finish!!")

