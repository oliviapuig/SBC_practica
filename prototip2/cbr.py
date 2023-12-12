#from typing import Any
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import DistanceMetric
import numpy as np
#from sklearn.preprocessing import OneHotEncoder
#from utils import Usuari
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors

class CBR:
    def __init__(self, cases, clustering, books): #cases és el pandas dataframe de casos
        self.cases = cases
        self.clustering = clustering
        self.books = books
    
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
        cl = self.clustering.predict(vector)[0]
        veins = self.cases[self.cases.cluster == cl]
        
        distancies = veins['vector'].apply(lambda x: np.linalg.norm(vector - np.array(list(x)), axis=1)) # distancia euclidea 
        veins_ordenats = sorted(((index, distancia) for index, distancia in enumerate(distancies)), key=lambda x: x[1])

        return veins_ordenats[:5] if len(veins_ordenats) >= 5 else veins_ordenats
    
    def reuse(self, user, users):
        
        # users és una llista de tuples (usuari, similitud)
        print(users)
        """
        Retorna els 3 llibres que més haurien d'agradar a l'usuari segons un KNN dels usuaris
        """
        # Cogemos los vectores de los libros recomendados por los usuarios similares
        vector_llibres_recom = []
        book_ids = []
        for u, _ in users:
            for llibre in self.cases.iloc[u]['llibres_recomanats']:
                v_llibre = list(self.books[self.books.book_id == int(llibre)]['vector'])[0]
                b_id = int(self.books[self.books.book_id == int(llibre)]['book_id'].iloc[0])
                vector_llibres_recom.append(v_llibre)
                book_ids.append(b_id)
        # El resultado deberian ser 15 vectores de 85 elementos
        vector_user = user.vector.reshape(1,-1)
        vector_llibres_recom = np.array(vector_llibres_recom)

        # Hacemos un KNN con los vectores de los libros recomendados
        knn = NearestNeighbors(n_neighbors=3)
        knn.fit(vector_llibres_recom)
        distancias, indices = knn.kneighbors(vector_user)

        # Guardamos los book_ids de los libros más cercanos
        llibres_recom = []
        for i in indices[0]:
            llibres_recom.append(book_ids[i])


        # FALTA MIRAR Q LLIBRES SIGUIN DIFERENTS


        return llibres_recom
    
    def revise(self, user, llibres):
        """
        Ens quedem amb els 3 llibres amb més puntuació i eliminem puntuacions        
        Mirem la columna de clustering dels 3 llibres recomanats i calculem la similitud de l'usuari amb els llibres del cluster
        Si la similitud entre l'usuari i un llibre és superior a la de l'usuari i un dels llibres recomanats, intercanviem els llibres
        """
        user["llibres_recomanats"].append(llibres)
        for llibre in llibres:
            cluster = self.books[self.books.book_id==int(llibre)]["cluster"]
            # Coger todos los libros que coincidan con el cluster del libro recomendado
            llibres_del_cluster = self.books[self.books['cluster'] == cluster]
            for ll in llibres_del_cluster:
                if self.similarity(user, ll, "cosine") > self.similarity(user, llibre, "cosine"):
                    llibres[llibres.index(llibre)] = ll
                    break
        user["llibres_recomanats"] = llibres
    
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
        ll = self.reuse(user, users)
        user = self.revise(user, ll)
        user = self.review(user)
        self.retain(user)
        return user