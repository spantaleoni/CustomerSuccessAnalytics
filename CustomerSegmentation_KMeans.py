#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 13:42:19 2024
CUSTOMER SEGMENTATION with KMEANS CLUSTERING

@author: simonlesflex
"""

import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


data = {
        'Age': [22, 34, 78, 56, 35, 42, 62, 54, 24, 18, 65, 72, 43, 48, 49],
        'Annual Income (kEur)': [35, 67, 45, 78, 88, 45, 55, 48, 50, 64, 100, 120, 45, 56, 60],
        'Spending Propensity Score (1-100)': [30, 55, 35, 45, 50, 60, 45, 50, 80, 25, 30, 35, 45, 90, 20]
        }

DataDF = pd.DataFrame(data)

# Usin KMeans methodology for clustering
KmeansDF = KMeans(n_clusters=3, random_state=42)
DataDF['Cluster'] = KmeansDF.fit_predict(DataDF[['Age', 'Annual Income (kEur)', 'Spending Propensity Score (1-100)' ]])

plt.figure(figsize=(20, 12))
colors = ['red', 'green', 'blue']

for i in range(3):
    plt.scatter(DataDF[DataDF['Cluster'] == i]['Age'],
                DataDF[DataDF['Cluster'] == i]['Annual Income (kEur)'],
                label=f'Cluster {i+1}',
                c=colors[i])

plt.title('Customer Segmentation')
plt.xlabel('Age')
plt.ylabel('Annual Income (kEur)')
plt.legend()