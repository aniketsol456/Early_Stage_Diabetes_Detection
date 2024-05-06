import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv("/content/diabetes_data_upload.csv")
dataset

#Features counting
Originalfeatures = dataset.columns
print("Originalfeatures =",len(Originalfeatures))

print("Originalfeatures =",Originalfeatures)

print("Dataset Shape: ",dataset.shape)

#1.Decision-Tree Algorithm
dataset1 = pd.get_dummies(dataset)
dataset1

print(dataset1.columns)

#separate features  and labels ______________Labels are nothing but actual value
labels = np.array(dataset1['Age'])
features = dataset1.drop('Age', axis=1) # 1 meanns row by default and 0 meanns col
features = np.array(features)

print("Features shape: ",features.shape)

from sklearn.model_selection import train_test_split
train_features,test_features,train_labels,test_labels = train_test_split(features,labels,test_size=0.20)

#training the model
from  sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_estimators=500)

rf.fit(train_features,train_labels)

prediction = rf.predict(test_features)
prediction

errors = abs(prediction - test_labels)
errors

#mean absolute error
mae=(errors/test_labels)
mae

#Mapping a accuracy score
accuracy = (1 - np.mean(mae))*100
accuracy

#For visulization of tree
from sklearn.tree import export_graphviz
import pydot
tree = rf.estimators_[8]
export_graphviz(tree,out_file='tree.dot')
(graph,) = pydot.graph_from_dot_file('tree.dot')
graph.write_png('tree.png')

