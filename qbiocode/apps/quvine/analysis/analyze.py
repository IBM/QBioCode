# Copyright 2026, IBM Corporation.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

from scipy.linalg import orthogonal_procrustes
from numpy.linalg import svd, eigh
from sklearn.cross_decomposition import CCA 
from scipy.spatial.distance import pdist, squareform 
from scipy.stats import spearmanr
from sklearn.neighbors import NearestNeighbors

def normalize(embedding, eps=1e-8): 
    mean = embedding.mean(axis=0, keepdims=True)
    std = embedding.std(axis=0, keepdims=True) + eps
    return (embedding - mean) / std

def procrustes_residual(Z1,Z2): 
    scaledZ1 = Z1 - Z1.mean(0)
    scaledZ2 = Z2 - Z2.mean(0)
    R, _ = orthogonal_procrustes(scaledZ1, scaledZ2)
    residual = np.linalg.norm(scaledZ1 - scaledZ2@R)/np.linalg.norm(scaledZ1)
    
    return residual

def cca_correlation(Z1, Z2, n_components=10):
    scaledZ1 = normalize(Z1)
    scaledZ2 = normalize(Z2)
    cca = CCA(n_components=n_components)
    Z1_c, Z2_c = cca.fit_transform(scaledZ1, scaledZ2)
    corrs = [np.corrcoef(Z1_c[:,i], Z2_c[:,i])[0,1] for i in range(n_components)]
    return np.mean(corrs)

def rsa_corr(Z1, Z2):
    D1 = pdist(Z1)
    D2 = pdist(Z2)
    rsa_corr = spearmanr(D1, D2).correlation
    return rsa_corr

def knn_sets(Z, k=10): 
    neighbors = NearestNeighbors(n_neighbors=k+1).fit(Z)
    idx = neighbors.kneighbors(Z, return_distance=False)
    return [set(row[1:]) for row in idx]

def knn_overlap(Z1, Z2, k=10):
    K1 = knn_sets(Z1, k)
    K2 = knn_sets(Z2, k)
    overlap = np.mean([len(K1[i] & K2[i])/k for i in range(len(K1))])
    return overlap

def effective_rank(s):
    p = s / s.sum()
    return np.exp(-np.sum(p * np.log(p + 1e-12)))

def plot_singular_values(singular_values, label='concatenate', filename=None):
    plt.figure()
    plt.plot(list(range(1, len(singular_values)+1)), [np.log(x) for x in singular_values], marker='o', color='blue', label=label)
    plt.xlabel('Singular Value Index')
    plt.ylabel('Log(Singular Value)')
    plt.legend()
    plt.show()
    if filename is not None:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    
def spectral_info(embeddings, labels, plot_flag=False):
    
    singular_values = []
    ks = []
    for z in embeddings:
        scaledZ = normalize(z)
        s = np.linalg.svd(scaledZ, compute_uv=False)
        singular_values.append(s)
        ks.append(len(s))
    k = min(ks)
    sk = []
    for s in singular_values:
        sk.append(s[:k])

    if plot_flag:
        fig = plt.figure()
        for i,s in enumerate(sk):
            plt.plot(np.log(s), label=labels[i])
        plt.xlabel('singular value index i')
        plt.ylabel('log($s_i$)')
        plt.legend()
        plt.show()
        fig.savefig('log_spectrum.png', 
                    dpi=300, 
                    bbox_inches='tight')
        
        fig = plt.figure()
        for i,s in enumerate(sk):
            plt.loglog(s, label=labels[i])
        plt.xlabel('singular value index i')
        plt.ylabel('loglog($s_i$)')
        plt.legend()
        plt.show()
        fig.savefig('loglog_spectrum.png', 
                    dpi=300, 
                    bbox_inches='tight')
        skn = [] 
        for s in sk:
            skn.append(s / np.sum(s))
        fig = plt.figure()
        for i,s in enumerate(skn):
            plt.plot(np.log(s), label=labels[i])
        plt.xlabel('singular value index i')
        plt.ylabel('log(normalized $s_i$)')
        plt.legend()
        plt.show()
        fig.savefig('log_normalized_spectrum.png', 
                    dpi=300, 
                    bbox_inches='tight')
    effective_ranks = {} 
    for i,label in enumerate(labels): 
        effective_ranks[label] = effective_rank(sk[i]) 
    
    return effective_ranks  
