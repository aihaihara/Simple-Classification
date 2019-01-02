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

#open your json file
with open("test2.json") as data: 
    data = json.load(data)

#framing your data
df = pandas.DataFrame.from_dict(data)
#print(df)

#change label to numerical data
factor = pandas.factorize(df['label'])
#print(factor)
df.label = factor[0]
definition = factor[1]
#print(definition)

#split training and testing data
train_x, test_x, train_y, test_y = train_test_split(df['text'], df['label'], test_size = 0.40, random_state = 21)

# #use countvectorizer as feature
# count_vect = CountVectorizer(analyzer='char',ngram_range=(2,3), max_features=2000)
# count_vect.fit(df['text'])
# train_x =  count_vect.fit_transform(train_x)
# test_x =  count_vect.transform(test_x)

#tfidf as feature
tfidf_vect_ngram = TfidfVectorizer(analyzer='char', ngram_range=(2,3), token_pattern=r'\w{1,}', max_features=2000)
tfidf_vect_ngram.fit(df['text'])
train_x =  tfidf_vect_ngram.transform(train_x)
test_x =  tfidf_vect_ngram.transform(test_x)

#call train the model
classifier = linear_model.LogisticRegression()
classifier.fit(train_x, train_y)

#test the model
pred_y = classifier.predict(test_x)

#for converting the numerical label to actual label
reversefactor = dict(zip(range(3), definition))
test_y = numpy.vectorize(reversefactor.get)(test_y)
pred_y = numpy.vectorize(reversefactor.get)(pred_y)

#print metric
accuracy = accuracy_score(test_y, pred_y)
score = precision_recall_fscore_support(test_y, pred_y, average='macro')
print('accuracy score:', accuracy)
print('precision, recall, and f1 score respectively:', score)

#print confusion matrix
report = classification_report(test_y, pred_y)
print(pandas.crosstab(test_y, pred_y, rownames=['Actual Label'], colnames=['Predicted Label']))
print(report)

#writing model into a file that can be used later
model = open("regression.pkl", 'wb')
pickle.dump(classifier, model)
model.close()
print("finish!!")

