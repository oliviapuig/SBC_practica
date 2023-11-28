from pandas import DataFrame
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Fem el clustering
def cluster_dataframe():
    '''
    Retorna el clustering del df de llibres
    '''
    with open('../data/books_clean.csv', 'rb') as file:
        df = pd.read_csv(file, sep=';')
    # Eliminem les columnes que no ens interessen
    df = df.drop(columns=['bookID', 'isbn', 'isbn13', 'title', 'authors', 'average_rating', 'language_code', 'ratings_count', 'work_ratings_count', 'work_text_reviews_count', 'ratings_1', 'ratings_2', 'ratings_3', 'ratings_4', 'ratings_5', 'image_url', 'small_image_url'])

    # Normalitzem les dades
    df_norm = (df - df.mean()) / (df.max() - df.min())

    return KMeans(n_clusters=5, random_state=0).fit(df)


def cluster_book(bookID):
    '''
    Retorna el cluster al que pertany un llibre
    '''
    return KMeans.labels_[bookID]


def get_cluster_books(cluster):
    '''
    Retorna una llista amb tots els llibres del cluster
    '''
    cluster_books = []
    for i in range(len(KMeans.labels_)):
        if KMeans.labels_[i] == cluster:
            cluster_books.append(i)
    return cluster_books