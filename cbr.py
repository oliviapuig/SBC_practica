from typing import Any
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import DistanceMetric
import numpy as np
from sklearn.preprocessing import OneHotEncoder

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
    
    def transform_user_to_numeric(self, encoder, users):
        print(users)
        for user in users:
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
    
    def reuse(self, user, users):
        """
        Agafar tots els llibres dels 5 usuaris més similars
        """
        for u in users:
            user.llibres_recomanats += u.llibres_recomanats
            user.puntuacio_llibres += u.puntuacio_llibres
        return user
    
    def revise(self, user):
        """
        Ens quedem amb els 3 llibres amb més puntuació i eliminem puntuacions        
        """
        user.llibres_recomanats = [x for _,x in sorted(zip(user.puntuacio_llibres, user.llibres_recomanats), reverse=True)][:3]
        user.puntuacio_llibres = []
        return user
    
    def retain(self, user):
        """
        Calculem la similitud de cosinus i, si es tracta d'un cas diferent, l'afegim a la bossa de casos
        """
        similarities = []
        for u in self.users:
            a = self.similarity(user, u, 'cosine')
            similarities.append(a)
        if np.average(similarities) <= 0.6:
            self.users.append(user)

    def recomana(self, user):
        user = self.transform_user_to_numeric(self.encoder, [user])[0]
        users = self.retrieve(user, "cosine")
        user = self.reuse(user, users)
        user = self.revise(user)
        self.retain(user)
        return user.llibres_recomanats