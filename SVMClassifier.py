# csv data file link ->  https://www.dropbox.com/s/hcrxfu9ztfpbsrw/irage_dataset.csv?dl=0

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn import preprocessing, cross_validation, svm
from sklearn.metrics import confusion_matrix, accuracy_score

class SVMClassifier:
    def __init__(self, filename = 'irage_dataset.csv'):
        self.df = pd.read_csv('irage_dataset.csv')
        
    def train_test_data(self, train_test_ratio = 0.15, input_data = 200000, label="fut_direction"):
        shuffled_data = self.df.sample(frac = 1)
        test_input = shuffled_data[1:input_data + 1]
#         print test_input.head()
        test_input.fillna(value=-99999, inplace=True)
        x = test_input.drop(["date", "fut_spread", "fut_direction", "30secAhead", "1minAhead"], 1)
        y = test_input[label]
        x = preprocessing.scale(x)
        x_train, x_test, y_train, y_test = cross_validation.train_test_split(x, y, test_size=0.1)
        return x_train, x_test, y_train, y_test
    
    def linear_svc(self, c = 1):
        x_train, x_test, y_train, y_test = self.train_test_data(input_data = len(self.df)) #len(self.df)
#         classifier = svm.SVC(kernel="linear", C=c).fit(x_train, y_train)
        classifier = OneVsRestClassifier(LinearSVC(random_state=0)).fit(x_train, y_train)
        class_pred = classifier.predict(x_test)
        return confusion_matrix(y_test, class_pred), accuracy_score(y_test, class_pred)
    
    def svm_rbf_model(self, gamma=0.8, c = 1):
        x_train, x_test, y_train, y_test = self.train_test_data(input_data = len(self.df))
        svc_rbf = svm.SVC(kernel="rbf", gamma = gamma , C=c).fit(x_train, y_train)
        rbf_pred = svc_rbf.predict(x_test)
        conf_mat = confusion_matrix(y_test, rbf_pred)
        acc_score = accuracy_score(y_test, rbf_pred)
        return conf_mat, acc_score
    
    def svm_poly_model(self, degree = 3, c=1):
        x_train, x_test, y_train, y_test = self.train_test_data(input_data = len(self.df))
        svc_poly = svm.SVC(kernel="poly", degree=3, C=c).fit(x_train, y_train)
        poly_pred = svc_poly.predict(x_test)
        conf_mat = confusion_matrix(y_test, poly_pred)
        acc_score = accuracy_score(y_test, poly_pred)
        return conf_mat, acc_score

svm_classifier = SVMClassifier()

start_time = time.time()
l_c_m, l_a_s = svm_classifier.linear_svc(c = 0.8)
print l_c_m
print l_a_s
linear_time = time.time()
print ("%s seconds --> linear svm for 10000 input data " %(linear_time - start_time)  )

start_time = time.time()
l_c_m, l_a_s = svm_classifier.svm_rbf_model(c = 0.8)
print l_c_m
print l_a_s
linear_time = time.time()
print ("%s seconds --> linear svm for 10000 input data " %(linear_time - start_time)  )

start_time = time.time()
l_c_m, l_a_s = svm_classifier.svm_poly_model(c = 0.8)
print l_c_m
print l_a_s
linear_time = time.time()
print ("%s seconds --> linear svm for 10000 input data " %(linear_time - start_time)  )
