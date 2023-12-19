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

#estableixo nou cas com el primer borrant els seus llibres recomanats (temporal)
casos_nous = cases.iloc[:102]
cases = cases.iloc[102:]
llibres_recomanats_baseline = casos_nous.llibres_recomanats
puntuacions_llibres_baseline = casos_nous.puntuacions_llibres
for i,_ in casos_nous.iterrows():
    casos_nous.at[i,'llibres_recomanats'] = []
    casos_nous.at[i,'puntuacions_llibres'] = []

cbr = CBR(cases, clustering, books)

puntuacions = []
for i,cas in casos_nous.iterrows():
    print('Nou cas',i)
    recomanacio = cbr.recomana(cas)
    print('Recomanació final:')
    for llibre in recomanacio.llibres_recomanats:
        print(books[books.book_id==int(llibre)]['title'].iloc[0])
        #cbr.justificacio(cas)
    puntuacions.append(recomanacio.puntuacions_llibres)
    print('\n')

# Flatten puntuacions
puntuacions = np.array(puntuacions)
puntuacions = puntuacions.flatten()
puntuacions_s = [str(p) for p in puntuacions]

# Plot barplot of puntuacions x the puntuation y the frequency
import matplotlib.pyplot as plt
from collections import Counter
# Ordenar "1", "2", "3", "4", "5"
puntuacions_s = sorted(puntuacions_s)
# Contar frecuencias
puntuacions_s = Counter(puntuacions_s)
# Plot
plt.figure(figsize=(10,5))
plt.bar(puntuacions_s.keys(), puntuacions_s.values())
plt.title('Distribució de les puntuacions')
plt.xlabel('Puntuació')
plt.ylabel('Frequència')
plt.savefig('./plots/puntuacions_recomanacions.png')

# Print mean and std
print('Puntuació mitjana:', np.mean(puntuacions))
print('Desviació estàndard:', np.std(puntuacions))