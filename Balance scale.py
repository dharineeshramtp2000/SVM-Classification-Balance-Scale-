#Import the Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot

#Import the Dataset
Dataset = pd.read_csv("balance_scale_weka_dataset.csv")
X = Dataset.iloc[:,:-1]
y = Dataset.iloc[:,-1]

#Feature scaling the independent variable
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X = sc_X.fit_transform(X)

#Splitting the data into train and test
from sklearn.model_selection import train_test_split
X_train , X_test, y_train, y_test = train_test_split(X, y , test_size = 0.3, random_state=0, shuffle =True)

#Importing the SVM Classification model
from sklearn.svm import SVC
classifier = SVC(C = 10.0, kernel = 'rbf',decision_function_shape='onr')
classifier.fit(X_train, y_train)

#Predicting the output for our SGD Linear Model with the test set
y_pred = classifier.predict(X_test)

#Its time for evaluating our model.
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
acc_train = accuracy_score(y_train, classifier.predict(X_train))
f1_train = f1_score(y_train, classifier.predict(X_train), average= 'weighted')

print("Traing set results")
print("ACCURACY ---------------------->",acc_train)
print("F1 SCORE ---------------------->",f1_train)

#Now lets see how well is our model. So now lets evaluate with our test set
acc_test = accuracy_score(y_test, y_pred)
f1_test = f1_score(y_test, y_pred, average= 'weighted')

print("Test set results")
print("ACCURACY ---------------------->",acc_test)
print("F1 SCORE ---------------------->",f1_test)

#Now lets have our famous Confusion Matrix to visually understand.
cm = confusion_matrix(y_test,y_pred)
print(cm)
