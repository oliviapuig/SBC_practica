import numpy as np
from sklearn.cluster import AgglomerativeClustering
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram,linkage
import pandas as pd

def cluster(casos):
    vectors = [cas.vector for cas in casos]

    # Calcular la matriu de distàncies
    distàncies = pdist(vectors, metric='euclidean')

    # Convertir la matriu de distàncies a una matriu quadrada
    matriu_distàncies = squareform(distàncies)

    # Aplicar l'algoritme d'Agrupació Jeràrquica
    model = AgglomerativeClustering(n_clusters=None, distance_threshold=0)  # Trieu el llindar de distància adequat
    clusters = model.fit_predict(matriu_distàncies)
    return clusters

