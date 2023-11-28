#from typing import Any
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import DistanceMetric
import numpy as np
#from sklearn.preprocessing import OneHotEncoder
#from utils import Usuari
from sklearn.cluster import KMeans

class CBR:
    def __init__(self, cases): #cases és el nom del fitxer de casos
        self.cases = cases
    
    def __str__(self):
        for case in self.cases:
            print(cases[case])
        return ""
    
    def get_users(self):
        '''
        Retorna una llista amb tots els usuaris
        '''
        users = cases["user_id"].unique()
        return users

   
    def similarity(self, user1, user2, metric):
        if metric == "hamming":
            # Hamming distance
            dist = DistanceMetric.get_metric('hamming')
            return dist.pairwise([user1.vector], [user2.vector])[0][0]
        elif metric == "cosine":
            return cosine_similarity([user1.vector], [user2.vector])[0][0]
        
    def retrieve(self, user, metric):
        """
        Return 5 most similar users
        """
         
        similarities = []
        for u in self.users:
            similarities.append((u, self.similarity(user, u, metric)))
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:1]
    
    def reuse(self, users):
        """
        Agafar tots els llibres dels 5 usuaris més similars
        """
        llibres_recom = []
        puntuacions = []
        for u, sim in users:
            llibres_recom += u.llibres_recomanats #afegeix a la llista els llibres recomanats de l'usuari similar
            puntuacions += u.puntuacio_llibres #afegeix a la llista les puntuacions dels llibres recomanats de l'usuari similar
        return llibres_recom, puntuacions
    
    def revise(self, user, llibres_recom, puntuacions):
        """
        Ens quedem amb els 3 llibres amb més puntuació i eliminem puntuacions        
        """
        llibres = [x for _,x in sorted(zip(puntuacions, llibres_recom), reverse=True)][:3]
        user.llibres_recomanats += llibres
        return user
    
    def review(self, user):
        """
        L'usuari valora els tres llibres
        """
        for llibre in user.llibres_recomanats:
            while True:
                puntuacio = int(input(f"Quina puntuació li donaries a la recomanació del llibre {llibre}? (0-5) "))
                if puntuacio >= 0 and puntuacio <= 5 and isinstance(puntuacio, int):
                    break
                else:
                    print("La puntuació ha de ser un valor entre 0 i 5")
            user.puntuacio_llibres.append(puntuacio)
        return user
    
    def retain(self, user):
        """
        Calculem la similitud de cosinus i, si es tracta d'un cas diferent, l'afegim a la bossa de casos
        """
        similarities = []
        for u in self.users:
            a = self.similarity(user, u, 'cosine')
            similarities.append(a)
        print("Similitud mitjana entre l'usuari nou i els altres:", np.average(similarities))
        if np.average(similarities) <= 0.6:
            self.users.append(user)

    def recomana(self, user):
        users = self.retrieve(user, "cosine")
        ll, punt = self.reuse(users)
        user = self.revise(user, ll, punt)
        user = self.review(user)
        self.retain(user)
        return user