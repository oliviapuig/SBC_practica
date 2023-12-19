# # SBC - Pràctica CBR
# 
# - Aina Gomila
# - Ruth Parajó
# - Olivia Puig
# - Marc Ucelayeta

from cbr import CBR
import pickle
import pandas as pd
import numpy as np

cases =pd.read_pickle('./data/casos.pkl')
books = pd.read_pickle('./data/llibres.pkl')

with open('./data/clustering/model_clustering_casos.pkl', 'rb') as arxiu:
    clustering = pickle.load(arxiu)


cases.at[1, 'llibres_recomanats'] = []
cases.at[1, 'puntuacions_llibres'] = []
nou_cas = cases.iloc[1]
cases=cases.iloc[2:]

cbr = CBR(cases,clustering,books)

recomanacio = cbr.recomana(nou_cas)
