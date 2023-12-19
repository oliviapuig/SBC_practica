# # SBC - Pràctica CBR
# 
# - Aina Gomila
# - Ruth Parajó
# - Olivia Puig
# - Marc Ucelayeta

from cbr import CBR
import pickle
import pandas as pd
# Libreria para medir el tiempo
import time

cases =pd.read_pickle('./data/casos.pkl')
books = pd.read_pickle('./data/llibres.pkl')

with open('./data/clustering/model_clustering_casos.pkl', 'rb') as arxiu:
    clustering = pickle.load(arxiu)

temps_recomanacio = []

# Agafa els ultims 100 casos
casos_nous = cases.tail(100)

for i,_ in casos_nous.iterrows():
    casos_nous.at[i,'llibres_recomanats'] = []
    casos_nous.at[i,'puntuacions_llibres'] = []

# Llista de 10 a 1601 (amb salt de 10)
r = [x for x in range(10,1501,10)]

print("AVIS! Aquest script tarda uns 40 minuts en executar-se")

for recom in r:
    cases_preview = cases.iloc[:recom].copy()
    print("RECOM:", recom)
    cbr = CBR(cases_preview, clustering, books, 'automatic')
    try:
        temps_inici = time.time()
        for i,cas in casos_nous.iterrows():
            recomanacio = cbr.recomana(cas)
        temps_final = time.time()
        temps_recomanacio.append(temps_final-temps_inici)
    except:
        print("ERROR")
        temps_recomanacio.append(0)

# Plot del temps de recomanacio
import matplotlib.pyplot as plt
plt.figure(figsize=(10,5))
plt.plot(r, temps_recomanacio)
plt.xlabel('Nombre de casos')
plt.ylabel('Temps (s)')
plt.title('Temps de recomanacio')
plt.savefig('./plots/temps_recomanacio.png')