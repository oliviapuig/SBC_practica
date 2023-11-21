from typing import Any
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import DistanceMetric
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.cluster import KMeans


class CBR:
    def __init__(self, users): # users és una llista amb tots els casos (bossa de casos)
        self.encoder = self.get_encoder()
        self.users = self.transform_user_to_numeric(self.encoder, users)
    
    def __str__(self):
        for user in self.users:
            print(user)
        return ""
    
    def get_users(self):
        return self.users
    
    def get_encoder(self):
        categories = [
            ["realisme", "romanticisme", "naturalisme", "simbolisme", "modernisme", "realisme magico", "postmodernisme"],
            ["amor", "aventura", "terror", "fantasia", "ciencia ficcio", "historica", "filosofica", "psicologica", "social", "politica", "religiosa", "erotica", "humoristica", "costumista", "negra", "realista", "fantastica", "mitologica", "poetica", "satirica", "biografica", "epica", "didactica", "teatral", "lirica", "epistolar", "dramatica", "epica", "didactica", "teatral", "lirica", "epistolar", "dramatica"],
            ["baixa", "mitjana", "alta"],
            ["simples", "complexes"],
            ["baix", "mitja", "alt"],
            ["accio", "reflexio"],
            ["curta", "mitjana", "llarga"],
            ["actual", "passada", "futura"],
            ["baix", "mitja", "alta"]
        ]
        encoder = OneHotEncoder(categories=categories, sparse_output=False)
        encoder.fit([["realisme", "amor", "baixa", "simples", "baix", "accio", "curta", "actual", "baix"]])
        return encoder
    
    def transform_user_to_numeric(self, encoder, usuaris_a_transformar):
        llista_ususaris = []
        for user in usuaris_a_transformar:
            categorical_attributes = []
            numeric_attributes = []
            for key, value in user.attributes.items():
                if isinstance(value, str):
                    categorical_attributes.append(value)
                elif isinstance(value, int):
                    numeric_attributes.append(value/100)

            transformed_categorical_data = encoder.transform([categorical_attributes])
            combined_data = np.hstack((numeric_attributes, transformed_categorical_data[0]))

            user.vector = combined_data
            llista_ususaris.append(user)

        return llista_ususaris
    
    def similarity(self, user1, user2, metric):
        if metric == "hamming":
            # Hamming distance
            dist = DistanceMetric.get_metric('hamming')
            return dist.pairwise([user1.vector], [user2.vector])[0][0]
        elif metric == "cosine":
            return cosine_similarity([user1.vector], [user2.vector])[0][0]
        
    def __calculate_optimal_k(self, inertia, k_range):
        """
        Calcula el valor óptimo de K utilizando el método del codo automatizado.
        :param inertia: Lista de valores de inercia para diferentes valores de K.
        :param k_range: Rango de valores de K considerados.
        :return: Valor óptimo de K.
        """
        # Coordenadas del primer y último punto
        p1 = np.array([k_range[0], inertia[0]])
        p2 = np.array([k_range[-1], inertia[-1]])

        # Distancia de cada punto a la línea
        distances = []
        for k, iner in zip(k_range, inertia):
            pk = np.array([k, iner])
            line_vec = p2 - p1
            point_vec = pk - p1
            distance = np.abs(np.cross(line_vec, point_vec)) / np.linalg.norm(line_vec)
            distances.append(distance)

        # Encontrar el índice del valor máximo de la distancia
        optimal_k_index = np.argmax(distances)
        return k_range[optimal_k_index]
    
    def make_clustering(self):
        # Número de usuarios
        num_usuarios = 100

        # Cada usuario tiene un vector de 13 posiciones
        # Generamos datos aleatorios para simular estos vectores
        np.random.seed(0)
        user_vectors = np.random.uniform(-1, 1, (num_usuarios, 13))

        # Método del codo para determinar el número óptimo de clusters
        inertia = []
        k_range = range(1, 11)
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=0, n_init=10).fit(user_vectors)
            inertia.append(kmeans.inertia_)
        
        # Calcular el valor óptimo de K
        optimal_k = self.__calculate_optimal_k(inertia, k_range)

        # Realizar el clustering con el valor óptimo de K
        kmeans = KMeans(n_clusters=int(optimal_k), random_state=0, n_init=10).fit(user_vectors)
        return kmeans
        
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
            llibres_recom += u.llibres_recomanats
            puntuacions += u.puntuacio_llibres
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
                puntuacio = int(input(f"Quina puntuació li donaries al llibre {llibre}? (0-5) "))
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
        print(np.average(similarities))
        if np.average(similarities) <= 0.6:
            self.users.append(user)

    def recomana(self, user):
        user = self.transform_user_to_numeric(self.encoder, [user])[0]
        users = self.retrieve(user, "cosine")
        ll, punt = self.reuse(users)
        user = self.revise(user, ll, punt)
        user = self.review(user)
        self.retain(user)
        return user