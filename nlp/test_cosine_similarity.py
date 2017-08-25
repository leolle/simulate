# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse
from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import kneighbors_graph
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform

# create firms' vocabulary matrix, row = firm, column = words.
firms = 100
words = 14
product = np.random.randint(2, size=(firms, words))
product_sparse = sparse.csr_matrix(product)

product_similarities = cosine_similarity(product_sparse)

similarities = cosine_similarity(product_sparse)
print('pairwise dense output:\n {}\n'.format(product_similarities))

#also can output sparse matrices
similarities_sparse = cosine_similarity(product_sparse,dense_output=False)
print('pairwise sparse output:\n {}\n'.format(similarities_sparse))

plt.figure(figsize=(10, 4))
model = AgglomerativeClustering(linkage='average',
                                connectivity=None,
                                n_clusters=2)
model.fit(product_similarities)
#plt.scatter(similarities_sparse, c=model.labels_,
#            cmap=plt.cm.spectral)



np.fill_diagonal(product_similarities, 0)
dists = squareform(product_similarities)
linkage_matrix = linkage(dists, "single")
dendrogram(linkage_matrix, labels=str(list(range(100))))



# Generate sample data
n_samples = 1500
np.random.seed(0)
t = 1.5 * np.pi * (1 + 3 * np.random.rand(1, n_samples))
x = t * np.cos(t)
y = t * np.sin(t)


X = np.concatenate((x, y))
X += .7 * np.random.randn(2, n_samples)
X = X.T


plt.show()
