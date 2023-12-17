# # SBC - Pràctica CBR
# 
# - Aina Gomila
# - Ruth Parajó
# - Olivia Puig
# - Marc Ucelayeta

from cbr_ruth import CBR
import pickle
import pandas as pd

cases =pd.read_pickle('./data/casos.pkl')
books = pd.read_pickle('./data/llibres.pkl')

with open('./data/clustering/model_clustering_casos.pkl', 'rb') as arxiu:
    clustering = pickle.load(arxiu)

#estableixo nou cas com el primer borrant els seus llibres recomanats (temporal)
casos_nous = cases.iloc[:102]
cases = cases.iloc[102:]
for i,_ in casos_nous.iterrows():
    casos_nous.at[i,'llibres_recomanats'] = []
    casos_nous.at[i,'puntuacions_llibres'] = []

cbr = CBR(cases,clustering,books)

for i,cas in casos_nous.iterrows():
    print('Nou cas',i)
    recomanacio = cbr.recomana(cas)
    print('Recomanació final:')
    for llibre in recomanacio.llibres_recomanats:
        print(books[books.book_id==int(llibre)]['title'].iloc[0])
        #cbr.justificacio(cas)
    print('\n')
   


