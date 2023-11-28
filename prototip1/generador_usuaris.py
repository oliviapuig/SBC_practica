import pickle
import random
import matplotlib.pyplot as plt
from collections import Counter
import pandas as pd
from utils import Usuari

llibres=pd.read_csv('../data/books_clean.csv')

estils_literaris = ["Modernisme","Noucentisme","Surrealisme","Realisme","Romanticisme","Barroquisme","Simbolisme","Escola Mallorquina","Escola Valenciana","Escola Catalana"
]

temes_de_llibres = [
    'Amor','Aventura','Ciència-ficció','Fantasia','Misteri','Històric','Romàntic','Drama','Comèdia','Thriller','Horror','Filosofia','Autoajuda','Biografia','Poesia','Clàssic','Infantil','Jovenil','No-ficció','Crim','Viatges','Ciència','Humor','Religió','Esport','Política','Culinària','Terror','Ecològic','Educatiu','Futurista','Western','Espionatge','Art','Negoci','Familiar','Ficció històrica','Romanticisme','Conte de fades','Teatre','Mitologia','Ucronia','Apocalíptic','Ciberpunk','Distopia','Realisme màgic','Satira'
]

complexitat = ['baixa','mitja','alta']

demografia = ['infantil','juvenil','adult']

situacio= ['estudiant', 'treballador', 'atur', 'jubilitat']

estat_civil= ['solter', 'casat', 'divorciat', 'viudu']

llibres= list(llibres['isbn13'])



atributs = [estils_literaris,temes_de_llibres,complexitat,demografia,situacio,estat_civil,llibres]


probabilitats_normalitzades=[]
num_usuaris = 100
llistes_categories=atributs
# Bucle per cada llista de categories
for llista_categories in llistes_categories:
    # Generar probabilitats aleatòries per a cada categoria
    probs=[random.uniform(0.01, 1) for _ in range(len(llista_categories))]

    # Normalitzar les probabilitats perquè sumin 1
    probabilitats_normalitzades.append([prob / sum(probs) for prob in probs])
    
llista_usuaris=[]
for _ in range(num_usuaris):

    llibres_recomenats=[]
    llibres_usuari=[]

    for i in range(3): #afegim 3 llibres recomanats aleatoris en funció de la probabilitat assignada a cada isbn
        llibre_recomenat=random.choices(llistes_categories[6], weights=probabilitats_normalitzades[6])[0]
        while llibre_recomenat in llibres_recomenats: # si ja està dins la llista tria un altre
            llibre_recomenat=random.choices(llistes_categories[6], weights=probabilitats_normalitzades[6])[0]
        llibres_recomenats.append(llibre_recomenat)

    for i in range(5): # afegim 5 llibres aleatoris que s'ha llegit l'usuari  en funció de la probabilitat assignada a cada isbn
        llibre_usuari=random.choices(llistes_categories[6], weights=probabilitats_normalitzades[6])[0]
        while llibre_usuari in llibres_usuari or llibre_usuari in llibres_recomenats: # si ja està a la llista dels recomanats o llegits tria un altre
            llibre_usuari=random.choices(llistes_categories[6], weights=probabilitats_normalitzades[6])[0]
        llibres_usuari.append(llibre_usuari)

    '''llista_usuaris.append({'estils_literaris': random.choices(llistes_categories[0], weights=probabilitats_normalitzades[0])[0],
                       'temes_de_llibres': random.choices(llistes_categories[1], weights=probabilitats_normalitzades[1])[0],
                       'complexitat': random.choices(llistes_categories[2], weights=probabilitats_normalitzades[2])[0],
                       'demografia': random.choices(llistes_categories[3], weights=probabilitats_normalitzades[3])[0],
                       'situacio': random.choices(llistes_categories[4], weights=probabilitats_normalitzades[4])[0],
                       'estat_civil': random.choices(llistes_categories[5], weights=probabilitats_normalitzades[5])[0],
                       'llibres_usuari': llibres_usuari,
                       'val_llibres': random.sample(range(6), 3),
                       'llibres_recomanats': llibres_recomenats,
                       'puntuacio_llibres': random.sample(range(6), 3)
                       } )'''
    llista_usuaris.append({'llibres_usuari': llibres_usuari,
                       'val_llibres': random.choices(range(6),k= 5), #genera valoracions del 0 al 5 aleatories per cada llibre llegit
                       'llibres_recomanats': llibres_recomenats,
                       'puntuacio_llibres': random.choices(range(6),k= 5) #genera puntuacions del 0 al 5 aleatories per cada llibre recomanat
                       } )


'''for clau in llista_usuaris[0].keys():
    if clau != 'llibres_recomanats' and clau != 'puntuacio_llibres':
        # Obtenir els valors de la clau de tota la llista
        valors_clau = [diccionari[clau] for diccionari in llista_usuaris]
        
        # Comptar la freqüència de cada valor
        contador = Counter(valors_clau)
        # Crear un gràfic de barres amb les freqüències
        plt.bar(contador.keys(), contador.values())
        plt.xlabel(clau)
        plt.ylabel('Freqüència')
        plt.title(f'Distribució de {clau}')
        plt.xticks(rotation=45, ha='right')
        plt.show()'''


usuaris_instancies = []

# Iterar sobre la lista de diccionarios y crear instancias de Usuario
for i,usuari_atr in enumerate(llista_usuaris):
    usuari = Usuari(i,usuari_atr)
    usuaris_instancies.append(usuari)


with open('usuaris.pkl', 'wb') as file:
    pickle.dump(usuaris_instancies, file)
