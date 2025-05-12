# Customer-Segmentation-Using-KMeans
This project performs customer segmentation using KMeans clustering, a type of unsupervised machine learning, to group customers based on spending behavior and demographics. The aim is to help businesses tailor marketing strategies for distinct customer groups.

## 📄 Overview

The dataset includes the following features:

* **Gender**
* **Age**
* **Annual Income (k\$)**
* **Spending Score (1–100)**

These features are used to cluster customers based on similar purchasing behaviors.

---

## 🔍 Exploratory Data Analysis (EDA)

Before clustering, an in-depth EDA was performed:

* **Data Cleaning**: Confirmed no missing values
* **Distribution Analysis**: Histograms and box plots to explore Age, Income, and Spending Score
* **Correlation Analysis**: Heatmap to assess relationships between numerical variables

---

## 🧠 Machine Learning - Clustering

### 📌 Clustering Steps

1. **Feature Selection**: `Age`, `Annual Income`, and `Spending Score` were selected.
2. **Elbow Method**: Used to determine optimal number of clusters via Within-Cluster Sum of Squares (WCSS).

   ```python
   wcss = []
   for i in range(1,11):
       Kmeans = KMeans(n_clusters=i, init='k-means++', random_state=0)
       Kmeans.fit(X)
       wcss.append(Kmeans.inertia_)
   ```
3. **Model Training**:

   ```python
   kmeans_model = KMeans(n_clusters=5, init='k-means++', random_state=0)
   y_kmeans = kmeans_model.fit_predict(X)
   ```
4. **Customer Cluster Visualization**:

   ```python
   plt.scatter(X[y_kmeans == 0,0], X[y_kmeans == 0,1], s=80, c='red', label='customer 0')
   plt.scatter(X[y_kmeans == 1,0], X[y_kmeans == 1,1], s=80, c='blue', label='customer 1')
   ...
   plt.title('CLUSTERS OF CUSTOMERS')
   plt.xlabel('Annual Income (k$)')
   plt.ylabel('Spending Score (1-100)')
   plt.legend()
   ```

---

## 📊 Insights

* **Five distinct customer segments** were identified based on income and spending patterns.
* Some clusters represent **high-income, high-spending** customers, while others capture **low-income, low-spending** groups.
* There's a clear group of **young high spenders** who don’t earn much but spend heavily — possibly students or impulsive buyers.
* Middle-income customers split into two groups: conservative spenders vs. frequent shoppers.

---

## 🛠 Libraries Used

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from scipy import stats
import warnings
```

---

## 🚀 How to Use

1. Clone this repository.
2. Install the dependencies:

   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn
   ```
3. Open and run the Jupyter notebook:

   ```bash
   jupyter notebook "Customer Segmentation using KMeans.ipynb"
   ```

---

## 📁 Files

* `Customer Segmentation using KMeans.ipynb` – Main notebook containing all code, visualizations, and insights.
* `README.md` – Project overview and documentation.
