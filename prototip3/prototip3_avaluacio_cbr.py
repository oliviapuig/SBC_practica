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

tipus = 'automatic'

casos_nous = cases.iloc[:102]
cases = cases.iloc[102:]
llibres_recomanats_baseline = casos_nous['llibres_recomanats'].copy()
puntuacions_llibres_baseline = casos_nous['puntuacions_llibres'].copy()
for i,_ in casos_nous.iterrows():
    casos_nous.at[i,'llibres_recomanats'] = []
    casos_nous.at[i,'puntuacions_llibres'] = []

cbr = CBR(cases, clustering, books,tipus)

puntuacions = []
llibres_recomanats = []
puntuacions_llibres = []
for i,cas in casos_nous.iterrows():
    print('Nou cas',i)
    recomanacio = cbr.recomana(cas)
    puntuacions.append(recomanacio.puntuacions_llibres)
    llibres_recomanats.append(recomanacio.llibres_recomanats)
    puntuacions_llibres.append(recomanacio.puntuacions_llibres)
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

# Guardar els llibres baseline i recomanats
l_recomanats_baseline = []
for llibres in llibres_recomanats_baseline:
    clusters = []
    for i, llibre in enumerate(llibres):
        if i != 2:
            cluster = books[books.book_id==int(llibre)].iloc[0]
            clusters.append(cluster)
    l_recomanats_baseline.append(clusters)

l_recomanats = []
for llibres in llibres_recomanats:
    clusters = []
    for i, llibre in enumerate(llibres):
        if i != 2:
            cluster = books[books.book_id==int(llibre)].iloc[0]
            clusters.append(cluster)
    l_recomanats.append(clusters)

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import DistanceMetric
def similarity(user, case, metric):
    # case haurà de ser el num de cas dins dataframe
    if metric == "hamming":
        # Hamming distance
        dist = DistanceMetric.get_metric('hamming')
        return dist.pairwise(user.vector.reshape(1,-1), np.array(case.vector).reshape(1,-1))[0][0]
    elif metric == "cosine":
        user_vector_np = np.array(user.vector).reshape(1, -1)
        case_vector_np = np.array(case.vector).reshape(1,-1)
        return cosine_similarity(user_vector_np, case_vector_np)[0][0]

mean_sim = []
mean_full = []
for i in range(len(llibres_recomanats)):
    # Calculamos la similitud de coseno
    sim_1_1 = similarity(l_recomanats_baseline[i][0], l_recomanats[i][0], "cosine")
    sim_1_2 = similarity(l_recomanats_baseline[i][0], l_recomanats[i][1], "cosine")
    sim_2_1 = similarity(l_recomanats_baseline[i][1], l_recomanats[i][0], "cosine")
    sim_2_2 = similarity(l_recomanats_baseline[i][1], l_recomanats[i][1], "cosine")
    mean = (sim_1_1 + sim_1_2 + sim_2_1 + sim_2_2) / 4
    mean = np.max([sim_1_1, sim_1_2, sim_2_1, sim_2_2])
    mean_full.append(np.array([sim_1_1, sim_1_2, sim_2_1, sim_2_2]))
    mean_sim.append(mean)

mean_sim = np.array(mean_sim)
mean_full = np.array(mean_full)

print()
print("Mitjana de similitud:", np.mean(mean_sim))


