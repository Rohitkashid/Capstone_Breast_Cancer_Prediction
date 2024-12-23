from __future__ import print_function
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn import tree
import warnings
warnings.filterwarnings('ignore')
df = pd.read_csv("cancer.csv")
df.replace('?',-99999,inplace=True)
df.drop(columns=['id'], inplace=True)
df.dtypes
df['classes'].value_counts()
sns.heatmap(df.corr(),annot=True)
features = df[['clump_thickness','unif_cell_size','unif_cell_shape','marg_adhesion','single_epith_cell_size','bare_nuclei','bland_chrom','norm_nucleoli','mitoses']]
target = df['classes']
labels = df['classes']
# Initializing empty lists to append all model's name and corresponding name
acc = []
model = []
from sklearn.model_selection import train_test_split
Xtrain, Xtest, Ytrain, Ytest = train_test_split(features,target,test_size = 0.2,random_state =42)
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score

DecisionTree = DecisionTreeClassifier(criterion="entropy",random_state=42,max_depth=5)

DecisionTree.fit(Xtrain,Ytrain)

predicted_values = DecisionTree.predict(Xtest)
x = metrics.accuracy_score(Ytest, predicted_values)
acc.append(x)
model.append('Decision Tree')
print("DecisionTrees's Accuracy is: ", x*100)

print()
print("classification_report: ")
print(classification_report(Ytest,predicted_values))
# Cross validation score (Decision Tree)
score =cross_val_score(DecisionTree, features, target,cv=5)
# score = cross_val_score(NaiveBayes,features,target,cv=5)
print(score)
from sklearn.naive_bayes import GaussianNB

NaiveBayes = GaussianNB()

NaiveBayes.fit(Xtrain,Ytrain)

predicted_values = NaiveBayes.predict(Xtest)
x = metrics.accuracy_score(Ytest, predicted_values)
acc.append(x)
model.append('Naive Bayes')
print("Naive Bayes's Accuracy is: ", x*100)

print()
print("classification_report: ")
print(classification_report(Ytest,predicted_values))
# Cross validation score (NaiveBayes)
score = cross_val_score(NaiveBayes,features,target,cv=5)
print(score)
from sklearn.svm import SVC

SVM = SVC(gamma='auto')

SVM.fit(Xtrain,Ytrain)

predicted_values = SVM.predict(Xtest)

x = metrics.accuracy_score(Ytest, predicted_values)
acc.append(x)
model.append('SVM')
print("SVM's Accuracy is: ", x*100)

print()
print("classification_report: ")
print(classification_report(Ytest,predicted_values))
# Cross validation score (SVM)
score = cross_val_score(SVM,features,target,cv=5)
print(score)
from sklearn.linear_model import LogisticRegression

LogReg = LogisticRegression(random_state=42)

LogReg.fit(Xtrain,Ytrain)

predicted_values = LogReg.predict(Xtest)

x = metrics.accuracy_score(Ytest, predicted_values)
acc.append(x)
model.append('Logistic Regression')
print("Logistic Regression's Accuracy is: ", x*100)

print()
print("classification_report: ")
print(classification_report(Ytest,predicted_values))
# Cross validation score (Logistic Regression)
score = cross_val_score(LogReg,features,target,cv=5)
print(score)
from sklearn.ensemble import RandomForestClassifier

RF = RandomForestClassifier(n_estimators=20, random_state=0)
RF.fit(Xtrain,Ytrain)

predicted_values = RF.predict(Xtest)

x = metrics.accuracy_score(Ytest, predicted_values)
acc.append(x)
model.append('RF')
print("RF's Accuracy is: ", x*100)

print()
print("classification_report: ")
print(classification_report(Ytest,predicted_values))
# Cross validation score (Random Forest)
score = cross_val_score(RF,features,target,cv=5)
print(score)
from sklearn.neighbors import KNeighborsClassifier

knn = []
for i in range(1, 21):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(Xtrain, Ytrain)

# Predicting the Test set results

Ypred = knn.predict(Xtest)

# Making the Confusion Matrix


x = metrics.accuracy_score(Ytest, predicted_values)
acc.append(x)
model.append('KNN')

print("Accuracy score of train KNN:-", x * 100)

print()
print("classification_report: ")
print(classification_report(Ytest, predicted_values))

print()
print("Confusion_matrixt: ")
KNN = confusion_matrix(Ytest, Ypred)
print(KNN)
plt.figure(figsize=[10,5],dpi = 100)
plt.title('Accuracy Comparison')
plt.xlabel('Accuracy')
plt.ylabel('Algorithm')
sns.barplot(x = acc,y = model,palette='dark')
accuracy_models = dict(zip(model, acc))
for k, v in accuracy_models.items():
    print (k, '-->', v)

import pickle
RF_pkl_filename = 'RandomForest.pkl'
# Open the file to save as pkl file
RF_Model_pkl = open(RF_pkl_filename, 'wb')
pickle.dump(RF, RF_Model_pkl)
# Close the pickle instances
RF_Model_pkl.close()

data = np.array([[5,3,3,3,2,9,4,4,1]])
prediction = knn.predict(data)
print(prediction)
if prediction == 0:
    print("Cancer is benign i.e not harmful")
else:
    print("Cancer is malignant")
data = np.array([[10,7,7,6,4,10,4,1,2]])
prediction = RF.predict(data)
print(prediction)

