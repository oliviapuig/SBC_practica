# # SBC - Pràctica CBR
# 
# - Aina Gomila
# - Ruth Parajó
# - Olivia Puig
# - Marc Ucelayeta

from cbr_uce import CBR
import pickle
import pandas as pd


cases =pd.read_pickle('./data/casos.pkl')
books = pd.read_pickle('./data/llibres.pkl')

with open('./data/clustering/model_clustering_casos.pkl', 'rb') as arxiu:
    clustering = pickle.load(arxiu)

#estableixo nou cas com el primer borrant els seus llibres recomanats (temporal)
cases.at[0, 'llibres_recomanats'] = []
cases.at[0, 'puntuacions_llibres'] = []
nou_cas = cases.iloc[0]
cases = cases.iloc[1:]

cbr = CBR(cases, clustering, books)

db_nou = cbr.inicia("prototip3/prova.txt")

for i in range(len(db_nou)):
    user = db_nou.iloc[i]
    recomanacio = cbr.recomana(user)
    print("Usuari final:")
    print('Recomanació final:')
    for llibre in recomanacio.llibres_recomanats:
        print(books[books.book_id==int(llibre)]['title'].iloc[0])