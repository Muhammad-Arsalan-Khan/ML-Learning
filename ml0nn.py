import pandas as pd 
import numpy as np 
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


data = {
    'Years of Experience': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'salary': [40000, 45000, 50000, 55000, 60000, 65000, 70000, 75000, 80000, 85000],
    'spending': [20000, 22000, 25000, 27000, 30000, 32000, 35000, 37000, 40000, 42000],
    'savings': [10000, 12000, 15000, 17000, 20000, 22000, 25000, 27000, 30000, 32000]
}

df = pd.DataFrame(data)
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df)

pca = PCA(n_components=2)
pca_data = pca.fit_transform(scaled_data)
variance_ratio = pca.explained_variance_ratio_
# print(f"Explained Variance Ratio: {np.round(variance_ratio * 100, 2)}")

pca_df = pd.DataFrame(data=pca_data, columns=['Principal Component 1', 'Principal Component 2'])

# print(pca_df)

plt.figure(figsize=(8, 6))
plt.scatter(pca_df['Principal Component 1'], pca_df['Principal Component 2'], c='blue', edgecolor='k')
plt.title('PCA of Salary Data')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2') 
plt.show()


