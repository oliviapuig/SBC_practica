#from typing import Any
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import DistanceMetric
import numpy as np
#from sklearn.preprocessing import OneHotEncoder
#from utils import Usuari
from sklearn.cluster import KMeans

class CBR:
    def __init__(self, cases, books): #cases és el pandas dataframe de casos
        self.cases = cases
        self.books = books
    
    def __str__(self):
        for case in self.cases:
            print(self.cases[case])
        for book in self.books:
            print(self.books[book])
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
            return dist.pairwise(user["vector"], self.cases[case]["vector"])[0][0]
        elif metric == "cosine":
            return cosine_similarity(user["vector"], self.cases[case]["vector"])[0][0]
        
    def retrieve(self, user, metric):
        """
        Return 5 most similar users
        """
        similarities = []
        for case in range(len(self.cases)):
            similarities.append((self.cases[case], self.similarity(user, case, metric)))
            # self.cases[case] perq volem posar el num de cas
            # (user, case, metric) perq volem fer similarity entre l'usuari i el cas, 
            # per tant necessitem el numero de cas per a la funcio similarity poder fer self.cases[case]
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:5]
    
    def reuse(self, users):
        # users és una llista de tuples (usuari, similitud)
        """
        Agafar tots els llibres dels 5 usuaris més similars
        """
        llibres_recom = []
        puntuacions = []
        for u, sim in users:
            llibres_recom += self.cases[u]['llibres_recomanats'] #afegeix a la llista els llibres recomanats de l'usuari similar
            puntuacions += self.cases[u]['puntuacions'] #afegeix a la llista les puntuacions dels llibres recomanats de l'usuari similar
        return llibres_recom, puntuacions
        
    def revise(self, user, llibres_recom, puntuacions):
        """
        Ens quedem amb els 3 llibres amb més puntuació i eliminem puntuacions        
        Mirem la columna de clustering dels 3 llibres recomanats i calculem la similitud de l'usuari amb els llibres del cluster
        Si la similitud entre l'usuari i un llibre és superior a la de l'usuari i un dels llibres recomanats, intercanviem els llibres
        """
        llibres = [x for _,x in sorted(zip(puntuacions, llibres_recom), reverse=True)][:3]
        user["llibres_recomanats"].append(llibres)
        for llibre in llibres:
            cluster = self.books[llibre]["cluster"]
            for ll in self.books:
                if self.books[ll]["cluster"] == cluster:
                    if self.similarity(user, ll, "cosine") > self.similarity(user, llibre, "cosine"):
                        llibres[llibres.index(llibre)] = ll
                        break
        user["llibres_recomanats"] = llibres

    def review(self, user):
        # user és un diccionari!!!
        """
        L'usuari valora els tres llibres
        """
        for llibre in user["llibres_recomanats"]:
            while True:
                puntuacio = int(input(f"Quina puntuació li donaries a la recomanació del llibre {llibre}? (0-5) "))
                if puntuacio >= 0 and puntuacio <= 5 and isinstance(puntuacio, int):
                    break
                else:
                    print("La puntuació ha de ser un valor entre 0 i 5")
            user["puntuacions"].append(puntuacio)
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
        users = self.retrieve(user, "cosine")
        ll, punt = self.reuse(users)
        user = self.revise(user, ll, punt)
        user = self.review(user)
        self.retain(user)
        return user