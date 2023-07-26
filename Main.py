#!/usr/bin/env python
# coding: utf-8


import pandas as pd
from gensim.models import Word2Vec
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import re


data=pd.read_csv('dataset.csv')


# Defining the default stopwords and lemmatizer
default_stopwords = set(stopwords.words('english'))

def preprocess_data(text):
    tokens = word_tokenize(text)
   
    # Removing punctuation and stopwords
    tokens = [re.sub(r'[^\w\s-]', '', token.lower()) for token in tokens]
    tokens = [token.lower() for token in tokens if token.lower() not in default_stopwords]
        
    # Joining back the preprocessed data
    preprocessed_text = ' '.join(tokens)
    
    return preprocessed_text

data['Preprocessed Term'] = data['Search Term'].apply(preprocess_data)
print(data.head())


#Training the model
preprocessed_texts=[text.split() for text in data['Preprocessed Term']]
model=Word2Vec(preprocessed_texts ,vector_size=100,window=5,min_count=1,workers=4)

#Converting text to vectors
data['Vector Terms']=data['Preprocessed Term'].apply(lambda text:[model.wv[word] for word in text.split() if word in model.wv])


#Computing the mean vector for each text
text_vectors = []
for vec_list in data['Vector Terms']:
    if len(vec_list) > 0:
        mean_vec = np.mean(vec_list, axis=0)
        text_vectors.append(mean_vec)

# Convert list of vectors to numpy array
text_vectors = np.array(text_vectors)


# Applying dimensionality reduction using PCA
pca = PCA(n_components=2)
pca_vectors = pca.fit_transform(text_vectors)

# Visualizing the reduced-dimensional vectors
plt.figure(figsize=(10, 6))
plt.scatter(pca_vectors[:, 0], pca_vectors[:, 1])
plt.title('PCA Visualization')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.savefig()

# Performing K-means clustering
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(pca_vectors)

# Visualizing the clusters in a 2D plot
plt.scatter(pca_vectors[:, 0], pca_vectors[:, 1], c=clusters, cmap='viridis')
plt.colorbar()
plt.title('Clusters')
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.savefig()

