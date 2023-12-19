# # SBC - Pràctica CBR
# 
# - Aina Gomila
# - Ruth Parajó
# - Olivia Puig
# - Marc Ucelayeta

from cbr_oliv import CBR
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
cases=cases.iloc[1:]

cbr = CBR(cases,clustering,books)

print(cases.loc[1])
print(cases.loc[1].vector)
recomanacio = cbr.recomana(nou_cas)
print('Recomanació final:')
for llibre in recomanacio.llibres_recomanats:
    print(books[books.book_id==int(llibre)]['title'].iloc[0])
