
#K-mean clustering

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load your dataset
data = pd.read_csv('/content/diabetes_data_upload.csv')
data

data.describe()

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Assuming 'class' is the target variable
X = data.drop('class', axis=1)
y = data['class']
print(X)
print(y)

# Convert categorical variables to numerical using one-hot encoding
X = pd.get_dummies(X)
X

# Standardize the data
scaler = StandardScaler()
# Fit the scaler to the data and transform the feature matrix X
X_scaled = scaler.fit_transform(X)
scaler

# Apply K-Means clustering
n_clusters = 2  # You can adjust this based on your requirements
kmeans = KMeans(n_clusters=n_clusters, random_state=32)
data['cluster'] = kmeans.fit_predict(X_scaled)
kmeans

# Visualize the clusters using a pair plot
sns.pairplot(data=data, hue='cluster', palette='viridis')
plt.title('Pair Plot of Clusters')
plt.show()

# Visualize the centroids in 2D (considering the first two features)
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=data['cluster'], cmap='viridis')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, marker='X', c='red')
plt.title('K-Means Clustering')
plt.xlabel('Age')
plt.ylabel('Gender')
plt.show()

# Visualize the distribution of a selected feature (e.g., 'Polyuria') within clusters
selected_feature = 'Polyuria'
plt.figure(figsize=(8, 6))
sns.countplot(x=selected_feature, hue='cluster', data=data, palette='viridis')
plt.title(f'Distribution of {selected_feature} within Clusters')
plt.xlabel(selected_feature)
plt.ylabel('Count')
plt.show()