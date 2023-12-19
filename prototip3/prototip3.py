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
import warnings
import random

warnings.filterwarnings("ignore")


cases =pd.read_pickle('./data/casos.pkl')
books = pd.read_pickle('./data/llibres.pkl')

with open('./data/clustering/model_clustering_casos.pkl', 'rb') as arxiu:
    clustering = pickle.load(arxiu)

index_cas=random.randint(0,len(cases)) #escollim nou cas de manera random
cases.at[index_cas, 'llibres_recomanats'] = []
cases.at[index_cas, 'puntuacions_llibres'] = []
nou_cas = cases.iloc[index_cas]
cases = cases.drop(index_cas)

cbr = CBR(cases,clustering,books)

recomanacio = cbr.recomana(nou_cas)
