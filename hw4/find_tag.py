
from __future__ import print_function

from sklearn.datasets import fetch_20newsgroups
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn import metrics

from sklearn.cluster import KMeans, MiniBatchKMeans

import logging
from optparse import OptionParser
import sys
from time import time

from sklearn.externals import joblib
import numpy as np
import pandas as pd
import warnings

DIR = sys.argv[1]
dataset = []
with open(DIR+"title_StackOverflow.txt", 'r+') as f:
    dataset = f.readlines()

print("Extracting features from the training dataset using a sparse vectorizer")
t0 = time()
vectorizer = TfidfVectorizer(max_df=0.5, max_features=10000,
                             min_df=2, stop_words='english',
                             use_idf=True)
#vectorizer = TfidfVectorizer(max_df=0.8, max_features=200000,
#                                 min_df=0.2, stop_words='english',
#                                 use_idf=True, tokenizer=tokenize_and_stem, ngram_range=(1,3))

X = vectorizer.fit_transform(dataset)

print("done in %fs" % (time() - t0))
print("n_samples: %d, n_features: %d" % X.shape)
print()

svd = TruncatedSVD(20)
normalizer = Normalizer(copy=False)
lsa = make_pipeline(svd, normalizer)

X = lsa.fit_transform(X)

km = KMeans(n_clusters=20, init='k-means++', max_iter=150, n_init=20,
                verbose=False)

print("Clustering sparse data with %s" % km)
t0 = time()
with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=DeprecationWarning)
    km.fit(X)
print("done in %0.3fs" % (time() - t0))
print()

print()


print("Top terms per cluster:")

original_space_centroids = svd.inverse_transform(km.cluster_centers_)
order_centroids = original_space_centroids.argsort()[:, ::-1]

tag = []
terms = vectorizer.get_feature_names()
for i in range(20):
    print("Cluster %d:" % i, end='')
    tag.append(terms[order_centroids[i, 0]])
    for ind in order_centroids[i, :10]:
        print(' %s' % terms[ind], end='')
        
    print()

print (tag)
joblib.dump(km, 'tag.mod')

test_data = np.genfromtxt(DIR+'check_index.csv', skip_header=1, delimiter=",")
print (test_data.shape)

tagging_table = np.zeros(shape=(20000, 20))
for i in range(len(dataset)):
    line = dataset[i].strip().lower().split()
    for j in range(20):
        if tag[j] in line:
            tagging_table[i, j] = 1


with open(sys.argv[2], 'w+') as f:
    f.write('ID,Ans\n')
    for i in range(test_data.shape[0]):
        a = int(test_data[i, 1])
        b = int(test_data[i, 2])
        tmp = [0 for j in range(20) if tagging_table[a, j]==1 and tagging_table[b, j]==1]
        if len(tmp)>0:
            f.write('%d,1\n' % i)
        else:
            f.write('%d,0\n' % i)
