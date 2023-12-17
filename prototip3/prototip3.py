# # SBC - Pràctica CBR
# 
# - Aina Gomila
# - Ruth Parajó
# - Olivia Puig
# - Marc Ucelayeta

from cbr import CBR
import pickle
import pandas as pd
import random
import numpy as np

cases =pd.read_pickle('./data/casos.pkl')
books = pd.read_pickle('./data/llibres.pkl')

with open('./data/clustering/model_clustering_casos.pkl', 'rb') as arxiu:
    clustering = pickle.load(arxiu)

#estableixo nou cas com el primer borrant els seus llibres recomanats (temporal)
'''cases.at[0, 'llibres_recomanats'] = []
cases.at[0, 'puntuacions_llibres'] = []
nou_cas = cases.iloc[0]
cases=cases.iloc[1:]'''

casos_nous = cases.iloc[:102]
cases = cases.iloc[102:]
for i,_ in casos_nous.iterrows():
    casos_nous.at[i,'llibres_recomanats'] = []
    casos_nous.at[i,'puntuacions_llibres'] = []

cbr = CBR(cases,clustering,books)

for i,cas in casos_nous.iterrows():
    print('Nou cas')
    recomanacio = cbr.recomana(cas)
    print("Usuari final:")
    print(recomanacio,'\n')

