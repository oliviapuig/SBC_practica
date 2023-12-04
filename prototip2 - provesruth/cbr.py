#from typing import Any
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import DistanceMetric
import numpy as np
#from sklearn.preprocessing import OneHotEncoder
#from utils import Usuari
from sklearn.cluster import KMeans

class CBR:
    def __init__(self, cases, clustering, books): #cases és el pandas dataframe de casos
        self.cases = cases
        self.clustering = clustering
        self.books=books
    
    def __str__(self):
        for case in self.cases:
            print(self.cases[case])
        return ""
    
    def get_users(self):
        '''
        Retorna una llista amb tots els usuaris
        '''
        users = self.cases["user_id"].unique()
        return users

    def similarity(self, user, case, metric):
        # case haurà de ser el num de cas dins dataframe
        if metric == "hamming":
            # Hamming distance
            dist = DistanceMetric.get_metric('hamming')
            return dist.pairwise(user.vector.reshape(1,-1), self.cases.iloc[case].vector.reshape(1,-1))[0][0]
        elif metric == "cosine":
            return cosine_similarity(user.vector.reshape(1,-1), self.cases.iloc[case].vector.reshape(1,-1))[0][0]
        
    def retrieve(self, user):
        """
        Return 10 most similar users
        """
        vector = user.vector.reshape(1,-1)
        cl=self.clustering.predict(vector)[0]
        veins = self.cases[self.cases.cluster == cl]
        #distancies = np.linalg.norm(user.vector - veins, axis=1)

        #veins_ordenats = veins[np.argsort(distancies)[1:]]  # Excluye el propio punto
        distancies = veins['vector'].apply(lambda x: np.linalg.norm(vector - np.array(list(x)), axis=1)) #distancia euclidea 
        # Encuentra el conjunto más cercano
        #veins_ordenats = veins.iloc[np.argmin(distancias)]
        # Suponiendo que distancias es una Serie de pandas
        veins_ordenats = sorted(((index, distancia) for index, distancia in enumerate(distancies)), key=lambda x: x[1])

        return veins_ordenats[:10] if len(veins_ordenats)>=10 else veins_ordenats
    
    def reuse(self, users):
        
        # users és una llista de tuples (usuari, similitud)
        """
        Retorna els 3 llibres que més haurien d'agradar a l'usuari
        """
        llibres_recom = []
        puntuacions = []
        for u, sim in users:
            llibres_recom += self.cases.iloc[u]['llibres_recomanats'] #afegeix a la llista els llibres recomanats de l'usuari similar
            puntuacions += self.cases.iloc[u]['puntuacions_llibres'] #afegeix a la llista les puntuacions dels llibres recomanats de l'usuari similar
        return llibres_recom, puntuacions
    
    def revise(self, user, llibres_recom, puntuacions):

        # user és un diccionari!!!
        # llibres_recom és una llista de llibres
        # puntuacions és una llista de puntuacions
        """
        Revisem la solució i mirem si pot haver algún llibre que li pot agradar més
        """
        llibres = [x for _,x in sorted(zip(puntuacions, llibres_recom), reverse=True)][:3]
        user['llibres_recomanats']+=llibres
        return user
    
    def review(self, user):

        # user és un diccionari!!!
        """
        L'usuari valora els tres llibres
        """
        for llibre in user['llibres_recomanats']:
            while True:
                puntuacio = int(input(f"Quina puntuació li donaries a la recomanació del llibre {self.books.loc[self.books[self.books['book_id'] == int(llibre)].index[0],'title']}? (0-5) ")) #agafo titol del llibre
                if puntuacio >= 0 and puntuacio <= 5 and isinstance(puntuacio, int):
                    break
                else:
                    print("La puntuació ha de ser un valor entre 0 i 5")
            user['puntuacions_llibres'].append(puntuacio)
        return user
    
    def retain(self, user):
        """
        Calculem la similitud de cosinus i, si es tracta d'un cas diferent, l'afegim a la bossa de casos
        """
        similarities = []
        for case in range(len(self.cases)):
            a = self.similarity(user, case, 'cosine')
            similarities.append(a)
        print("Similitud mitjana entre l'usuari nou i els altres:", np.average(similarities))
        if np.average(similarities) <= 0.6:
            self.cases.append(user, ignore_index=True)

    def recomana(self, user):
        # user es un diccionari!!!
        users = self.retrieve(user)
        ll, punt = self.reuse(users)
        user = self.revise(user, ll, punt)
        user = self.review(user)
        self.retain(user)
        return user