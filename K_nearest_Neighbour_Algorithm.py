
#3. K nearest neighbour(KNN)
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv("/content/diabetes_data_upload.csv")
dataset

dataset.head()

dataset1 = pd.get_dummies(dataset)
dataset1

X = dataset1.iloc[:,:-1].values
Y = dataset1.iloc[:,32].values
X
Y

#divide into train and split
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.20)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
X_train
X_test

from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=6)
classifier.fit(X_train,Y_train)

y_pred = classifier.predict(X_test)
y_pred

from sklearn.metrics import classification_report,confusion_matrix
print(classification_report(Y_test,y_pred))
print(confusion_matrix(Y_test,y_pred))
