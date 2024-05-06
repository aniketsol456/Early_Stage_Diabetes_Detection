
#2. == Navie Bayes Algorithm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv("/content/diabetes_data_upload.csv")
print("Dataset",dataset)

dataset = pd.get_dummies(dataset)
dataset

#separate features  and labels ______________Labels are nothing but actual value
labels = np.array(dataset['Age'])
features = dataset.drop('Age', axis=1) # 1 meanns row by default and 0 meanns col
features = np.array(features)

print("Feature Shape= ",features.shape)

X = dataset.iloc[:520,2:16].values
Y = dataset.iloc[:,16].values

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.20)
X_train

from sklearn.naive_bayes import BernoulliNB
classifierBNB=BernoulliNB()
classifierBNB.fit(X_train,Y_train)

from sklearn.linear_model import LogisticRegression
classifier=LogisticRegression()
classifier.fit(X_train,Y_train)

y_pred=classifier.predict(X_test)
print("Y_pred= ",y_pred)

from sklearn.metrics import confusion_matrix
confusion_matrix(Y_test,y_pred)

from sklearn.metrics import accuracy_score
accuracy_score(Y_test,y_pred)

# pos , neg = (Y==1).reshape(519,16) , (Y==0).reshape(519,16)
# plt.scatter(X[pos[:,0],0],X[pos[:,0],1],c="r",marker="+")
# plt.scatter(X[neg[:,0],0],X[neg[:,0],1],marker="o",s=10)
# plt.xlabel("Test 1")
# plt.ylabel("Test 2")
# plt.legend(["Yes","No"],loc=0)