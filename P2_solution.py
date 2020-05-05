import pandas as pd
import numpy as np
import pickle
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.externals import joblib
import sklearn.metrics as mt
import os
from optparse import OptionParser


#Train models with data and store models as pickle objects
def trainModel(features, labels):
	name = 'SVM'
	#split the data into training and testing sets
	train_feature, test_feature, train_class, test_class = train_test_split(features, labels, stratify=labels, test_size = 0.25)
	clf = Pipeline([('vect', CountVectorizer(stop_words="english")),
                      ('tfidf', TfidfTransformer()),
                      ('clf', LinearSVC()),])
	clf.fit(train_feature, train_class)
	j = joblib.dump(clf, 'SVM.pkl')
	print ("Training completed for "+name)
	prediction = clf.predict(test_feature)
	print('Classification Report for '+name)
	print(mt.classification_report(test_class, prediction))
	print('Confusion Matrix for '+name)
	print(pd.crosstab(test_class, prediction, rownames=['True'], colnames=['Predicted'], margins=True))   

def cross_val(features, labels):
	clf = Pipeline([('vect', CountVectorizer(stop_words="english")),
                      ('tfidf', TfidfTransformer()),
                      ('clf', LinearSVC()),])
	scores = cross_val_score(clf, features, labels, cv=10)
	print("{0} - Cross-validation scores: {1}".format('SVM', scores))
	print("Average cross-validation score: {:.2f}".format(scores.mean()))
 

def predictModel(input_sentence):
	input_sentence = [input_sentence]
	clf = joblib.load('SVM.pkl')
	print(clf.predict(input_sentence)[0])

        
if __name__ == '__main__':
	parser = OptionParser()
	parser.add_option("--mode", dest="mode", type="choice", choices=["train", "cross_val", "predict"], default="train")
	parser.add_option("--input", type="str", default="voting_data.csv")
	(options, args) = parser.parse_args()
	
	if options.mode in "train":
		assert options.input is not None
		data = pd.read_csv(options.input, encoding='utf-8')
		trainModel(data.text, data.label)
	elif options.mode  == "cross_val":
		assert options.input is not None
		data = pd.read_csv(options.input, encoding='utf-8')
		cross_val(data.text, data.label)
	elif options.mode == "predict":
		sentence = [options.input]
		predictModel(sentence[0])
	else:
		raise Exception("Invalid parser mode", options.mode)
        






    