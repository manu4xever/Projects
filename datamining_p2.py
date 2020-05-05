#Manu Gupta(1001599943)
import getopt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import graphviz
import sys
from IPython.display import display
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import export_graphviz
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.tree import DecisionTreeClassifier

from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(norm='l2', encoding='latin-1', ngram_range=(1, 2))#using tf-idf to classify and vectorize the feature
from sklearn.metrics import classification_report
from sklearn.externals import joblib
import pickle

#train function
def train(test_feature, test_class,train_feature, train_class):
    print("Test set score: {:.3f}".format(trained_model.score(test_feature, test_class)))# test score by using the trained model
    print("Training set score: {:.3f}".format(trained_model.score(train_feature, train_class)))#training score  by using the trained model
    prediction = trained_model.predict(test_feature)#preicting the test feature

    print("Confusion matrix:")#confusion matrix
    print(pd.crosstab(test_class, prediction, rownames=['True'], colnames=['Predicted'], margins=True))

    target_names = ['E', 'O', 'Others', 'V']
    print(classification_report(test_class, prediction, target_names=target_names))#classification report
    # prediction1('And as you know, nobody can reach the White House without the Hispanic vote.')

def crossval( features, labels):#cross-validation function
    trained_model = joblib.load('SVM.pkl')
    scores = cross_val_score(trained_model, features, labels, cv=10)#comparing values to get the best mean
    print("Cross-validation scores: {}".format(scores))
    print("Average cross-validation score: {:.2f}".format(scores.mean()))

def prediction1(str):#prediction function

    trained_model = joblib.load('SVM.pkl')#loading trained model
    pr_lab = trained_model.predict(str)
    print(pr_lab)#print the predicted label






if __name__ == "__main__":
    mode = sys.argv[2]#load mode from argument
    inp_file = sys.argv[4]#load the input from argument

    s = " ".join(sys.argv[4:])#expecting the last argument to be of filetype or else string-> for which all the argument will be joined after 'input:'
    s = s.replace("\\", '')#removing extra backslash
    PIK = "pickle.dat"  #creating a  pickle

    if 'predict' in mode:
        try:
            #load svm pickle model
            trained_model = joblib.load('SVM.pkl')
            inp_list = []
            inp_list.append(s)#append string to list
            with open(PIK, "rb") as f:#load pickle model"pickle.dat"->features
                inp_load =(pickle.load(f))

            pred_fit_vec = vectorizer.fit_transform(inp_load['text']).toarray()#vectorize the pickle object
            p = vectorizer.transform(inp_list)#transform the input
            prediction1(p)#call prediction function
        except:
            print ("train model first")
        sys.exit(1)

    df = pd.read_csv(inp_file, sep=',')
    # vectorizer = TfidfVectorizer(norm='l2', encoding='latin-1', ngram_range=(1, 2))
    features = vectorizer.fit_transform(df['text']).toarray()#extract featrures and give them a numeric id to each


    with open(PIK, "wb") as f:#storing the df as a pickle object
        pickle.dump(df, f)


    labels = df['label'] #extract labels of file
    train_feature, test_feature, train_class, test_class = train_test_split(
    features, labels, stratify=labels, random_state=0, train_size=.75, test_size=.25)#extracting train_feature, test_feature, train_class, test_class
    linearsvm = LinearSVC(random_state=0).fit(train_feature, train_class)#using linear svc model
    # store your trained model as pickle object
    joblib.dump(linearsvm, 'SVM.pkl')
    # load a pickle object
    trained_model = joblib.load('SVM.pkl')

    if 'train' in mode :
        print('training model')
        train(test_feature, test_class,train_feature, train_class)
    if 'cross_val' in mode:#call cross validation function
        crossval( features, labels)
