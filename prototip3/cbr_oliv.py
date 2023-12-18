from sys import displayhook
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import DistanceMetric
import numpy as np
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
import random

class CBR:
    def __init__(self, cases, clustering, books): #cases és el pandas dataframe de casos
        self.cases = cases
        self.clustering = clustering
        self.books=books
        self.iteracions = 0
    
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
            return dist.pairwise(user.vector.reshape(1,-1), np.array(case.vector).reshape(1,-1))[0][0]
        elif metric == "cosine":
            user_vector_np = np.array(user.vector).reshape(1, -1)
            case_vector_np = np.array(case.vector).reshape(1,-1)
            return cosine_similarity(user_vector_np, case_vector_np)[0][0]
    
          
    def retrieve(self, user):
        """
        Return 10 most similar users
        """
        vector = user.vector.reshape(1,-1)
        cl=self.clustering.predict(vector)[0]
        veins = self.cases[self.cases.cluster == cl]
        distancies = veins.apply(lambda x: self.similarity(user,x,'cosine'),axis=1)
        veins_ordenats = sorted(((index, distancia) for index, distancia in enumerate(distancies)), key=lambda x: x[1])

        return veins_ordenats[:5] if len(veins_ordenats)>=10 else veins_ordenats
        
    def reuse(self, user,users):
        
        # users és una llista de tuples (usuari, similitud)
        """
        Retorna els 3 llibres que més haurien d'agradar a l'usuari segons un KNN dels usuaris
        """
        # Cogemos los vectores de los libros recomendados por los usuarios similares
        vector_llibres_recom = []
        book_ids = []
        llibres_recomanats_afegits = set(user['llibres_recomanats']) # Conjunto de libros recomendados ya añadidos

         
        for u, _ in users:
            for llibre in self.cases.iloc[u]['llibres_recomanats']: # Cogemos los libros recomendados por el usuario
                if llibre not in llibres_recomanats_afegits:
                # si el libro no se ha leido el usuario actual y no se ha recomendado ya, continua, sino coge el siguiente libro
                    if llibre not in user['llibres_llegits']:
                        v_llibre = list(self.books[self.books.book_id == int(llibre)]['vector'])[0] # Cogemos el vector del libro
                        b_id = int(self.books[self.books.book_id == int(llibre)]['book_id'].iloc[0]) # Cogemos el id del libro
                        vector_llibres_recom.append(v_llibre) # Añadimos el vector a la lista de vectores
                        book_ids.append(b_id) # Añadimos el id a la lista de ids
                        llibres_recomanats_afegits.add(llibre)
                    
        # El resultado deberian ser 15 vectores de 85 elementos
        vector_user = user.vector.reshape(1,-1) # Cogemos el vector del usuario
        vector_llibres_recom = np.array(vector_llibres_recom) # Pasamos la lista de vectores a un array de numpy

        # Hacemos un KNN con los vectores de los libros recomendados
        knn = NearestNeighbors(n_neighbors=3)
        knn.fit(vector_llibres_recom)
        _, indices = knn.kneighbors(vector_user)

        # Guardamos los book_ids de los libros más cercanos
        llibres_recom = []
        for i in indices[0]:
            llibres_recom.append(book_ids[i])
        
        return llibres_recom
    
    def revise(self, user, llibres):
        """
        Ens quedem amb els 2 llibres amb més puntuació i eliminem puntuacions        
        Mirem la columna de clustering dels 3 llibres recomanats i calculem la similitud de l'usuari amb els llibres del cluster
        Si la similitud entre l'usuari i un llibre és superior a la de l'usuari i un dels llibres recomanats, intercanviem els llibres
        """
        for llibre in user["llibres_recomanats"]:
            cluster = self.books[self.books.book_id==int(llibre)]["cluster"]
            llibre_recomanat = self.books[self.books.book_id==int(llibre)].iloc[0]
            # Coger todos los libros que coincidan con el cluster del libro recomendado
            llibres_del_cluster = self.books[self.books['cluster'] == int(cluster.iloc[0])]
            for i, ll in llibres_del_cluster.iterrows():
                if ll['book_id'] not in user["llibres_recomanats"] and ll['book_id'] not in user["llibres_llegits"]:
                    inf len(user["llibres_recomanats"]) == 2:
                        llibre_recomanat = self.books[self.books.book_id == int(user["llibres_recomanats"][0])].iloc[0]
                        if self.similarity(user, ll, "cosine") > self.similarity(user, llibre_recomanat, "cosine"):
                            print('he entrat')
                            user["llibres_recomanats"][user["llibres_recomanats"].index(llibre)] = ll['book_id']
                            break
                    else:
                        # Recalcula i agafa el següent llibre més probable
                        user["llibres_recomanats"].append(ll['book_id'])
                        pass

        user["llibres_recomanats"] = user["llibres_recomanats"][:2]

        print("A continuació, et demanarem algunes preferències per millorar la recomanació.")

        # Preguntas de preferencias al usuario
        preferencia_llibre = input("¿Prefereixes llibres semblants als llegits o vols explorar? (Semblants/Explorar): ").lower()
        preferencia_popularitat = input("¿Prefereixes llibres populars (bestseller) o no tan populars? (Bestseller/No tan popular): ").lower()
        # Lógica para ajustar las recomendaciones según las preferencias del usuario
        if preferencia_llibre == 'semblants':
            # Aquí puedes incluir lógica para ajustar las recomendaciones según el género preferido del usuario
            print("Perfecte. Ajustant recomanacions semblants als llibres llegits...")

            # Calcula la similitud entre el usuario y los libros del mismo cluster
            distancies = llibres_del_cluster.apply(lambda x: self.similarity(user,x,'cosine'),axis=1)
            # añade a llibres recomanats el libro más similar del mismo cluster
            user["llibres_recomanats"].append(llibres_del_cluster.iloc[distancies.idxmax()]['book_id'])

        elif preferencia_llibre == 'explorar':
            # Aquí puedes incluir lógica para ajustar las recomendaciones según el género preferido del usuario
            print("Perfecte. Ajustant recomanacions més arriscades...")

            # Calcula el cluster de l'usuari actual
            cluster_usuari = user["cluster"].iloc[0]
            # calcula distancies entre clusters
            distancies_cluster = self.clustering.transform(user.vector.reshape(1,-1))
            # troba el cluster més proper al cluster de l'usuari actual
            cluster_mes_proper = distancies_cluster.argsort()[0][1]
            # afegir a llibres recomanats un llibre random del cluster més proper
            user["llibres_recomanats"].append(self.books[self.books['cluster'] == int(cluster_mes_proper)].sample(1)['book_id'].values[0])

        else:
            print("Opció no vàlida. Si us plau, respon 'semblants' o 'explorar'.")

        if preferencia_popularitat == 'bestseller':
            print("Perfecte. Ajustant recomanacions més populars...")
            # Añadir a llibres recomanats el libro más popular del cluster del usuario
            user["llibres_recomanats"].append(llibres_del_cluster[llibres_del_cluster['popular'] == True].sample(1)['book_id'].values[0])

        elif preferencia_popularitat == 'no tan popular':
            print("Perfecte. Ajustant recomanacions més arriscades...")
            # afegir a llibres recomanats un llibre random del cluster del usuari que sigui no popular
            user["llibres_recomanats"].append(llibres_del_cluster[llibres_del_cluster['popular'] == False].sample(1)['book_id'].values[0])

        else:
            print("Opció no vàlida. Si us plau, respon 'bestseller' o 'no tan popular'.")

        return user

    def revise(self, user):
        """
        Ens quedem amb els 2 llibres amb més puntuació i eliminem puntuacions        
        Mirem la columna de clustering dels 3 llibres recomanats i calculem la similitud de l'usuari amb els llibres del cluster
        Si la similitud entre l'usuari i un llibre és superior a la de l'usuari i un dels llibres recomanats, intercanviem els llibres
        """
        for llibre in user["llibres_recomanats"]:
            cluster = self.books[self.books.book_id==int(llibre)]["cluster"]
            llibre_recomanat = self.books[self.books.book_id==int(llibre)].iloc[0]
            llibres_del_cluster = self.books[self.books['cluster'] == int(cluster.iloc[0])]
            for i, ll in llibres_del_cluster.iterrows():
                if ll['book_id'] not in user["llibres_recomanats"] and ll['book_id'] not in user["llibres_llegits"]:
                    if len(user["llibres_recomanats"]) == 2:
                        llibre_recomanat = self.books[self.books.book_id == int(user["llibres_recomanats"][0])].iloc[0]
                        if self.similarity(user, ll, "cosine") > self.similarity(user, llibre_recomanat, "cosine"):
                            print('he entrat')
                            user["llibres_recomanats"][user["llibres_recomanats"].index(llibre)] = ll['book_id']
                            break
                    else:
                        # Recalcula i agafa el següent llibre més probable
                        user["llibres_recomanats"].append(ll['book_id'])
                        pass

        user["llibres_recomanats"] = user["llibres_recomanats"][:2]

        print("A continuació, et demanarem algunes preferències per millorar la recomanació.")

        # Preguntas de preferencias al usuario
        preferencia_llibre = input("¿Prefereixes llibres semblants als llegits o vols explorar? (Semblants/Explorar): ").lower()
        preferencia_popularitat = input("¿Prefereixes llibres populars (bestseller) o no tan populars? (Bestseller/No tan popular): ").lower()

        # Lògica per ajustar les recomanacions segons les preferències del usuari
        if preferencia_llibre == 'semblants':
            distancies = llibres_del_cluster.apply(lambda x: self.similarity(user, x, 'cosine'), axis=1)

            # Obté l'índex del llibre més semblant dins del cluster
            index_llibre_mes_semblant = distancies.idxmax()

            # Verifica la popularitat del llibre més semblant
            popularitat_llibre_mes_semblant = llibres_del_cluster.loc[index_llibre_mes_semblant, 'popular']

            # Filtra els llibres del cluster segons la preferència de popularitat de l'usuari
            if (preferencia_popularitat == 'bestseller' and popularitat_llibre_mes_semblant == True) or (preferencia_popularitat == 'no tan popular' and popularitat_llibre_mes_semblant == False):
                # Afegeix el llibre més semblant a la llista de llibres recomanats de l'usuari
                user["llibres_recomanats"].append(llibres_del_cluster.loc[index_llibre_mes_semblant, 'book_id'])
            else:
                print("El llibre més semblant no compleix amb la preferència de popularitat de l'usuari.")

        elif preferencia_llibre == 'explorar':
            cluster_usuari = user["cluster"].iloc[0]
            distancies_cluster = self.clustering.transform(user.vector.reshape(1, -1))
            cluster_mes_proper = distancies_cluster.argsort()[0][1]
            
            # Filtra els llibres del cluster més proper
            llibres_cluster_mes_proper = self.books[self.books['cluster'] == int(cluster_mes_proper)]

            # Filtra pels llibres que són bestsellers o no, segons la preferència de l'usuari
            if preferencia_popularitat == 'bestseller':
                llibres_recomanats_cluster_mes_proper = llibres_cluster_mes_proper[llibres_cluster_mes_proper['popular'] == True]
            elif preferencia_popularitat == 'no tan popular':
                llibres_recomanats_cluster_mes_proper = llibres_cluster_mes_proper[llibres_cluster_mes_proper['popular'] == False]
            else:
                print("Opció no vàlida per preferencia_popularitat. Si us plau, respon 'bestseller' o 'no tan popular'.")
                return user
            
            # Afegeix un llibre aleatori de la llista filtrada als llibres recomanats de l'usuari
            user["llibres_recomanats"].append(llibres_recomanats_cluster_mes_proper.sample(1)['book_id'].values[0])

        else:
            print("Opcions no vàlides. Si us plau, respon 'semblants' o 'explorar'.")

        return user
    
    def review(self, user):
        """
        L'usuari valora els tres llibres
        """
        for llibre in user['llibres_recomanats']:
            while True:
                print(f"Quina puntuació li donaries a la recomanació del llibre {self.books.loc[self.books[self.books['book_id'] == int(llibre)].index[0],'title']}? (0-5) ")
                puntuacio = random.randint(1,5)
                print(puntuacio)
                if puntuacio >= 0 and puntuacio <= 5 and isinstance(puntuacio, int):
                    break
                else:
                    print("La puntuació ha de ser un valor entre 0 i 5")
            user['puntuacions_llibres'].append(puntuacio)
        return user
    
    def retain(self, user,users):
        """
        calculem similitud entre casos SENSE TENIR EN COMPTE RECOMANACIONS
        - si cas molt diferent → ens el quedem
        - si el cas molt similar → mirem recomanacions
            - si les valoracions que ha posat a les recomanacions son totes 4<x<5 o 1<x<2 ens ho quedem perq casos extrems
            - si no, no ens el quedem perq cas similar
        - calcular utilitat
        """
        vector = user.vector.reshape(1,-1)
        cl=self.clustering.predict(vector)[0]
        veins = self.cases[self.cases.cluster == cl]

        similarities = []

        for case in range(len(veins)):
            a = self.similarity(user, self.cases.iloc[case], 'cosine')
            similarities.append(a)

        if np.average(similarities) <= 0.6:
            self.cases.loc[len(self.cases)] = user
        else:
            for i in range(len(user['puntuacions_llibres'])):
                if user['puntuacions_llibres'][i] > 2 or user['puntuacions_llibres'][i] < 4:
                    break
                elif i == len(user['puntuacions_llibres'])-1:
                    self.cases.append(user, ignore_index=True)
        
        self.utilitat(user,users) # actualitzem utilitat

    def utilitat(self, user,users):
        """
        Calcula la utilitat de l'usuari
        com calcular utilitat:
            - calculem utilitat al retain
            - cas utilitzat → llibre del cas és recomanat → llibre recomanat bona valoració → ENTENEM QUE EL CAS ÉS ÚTIL
            - un cas pot ser uttil de manera negativa
                - si el seu llibre es recomanat i rep una valoracio negativa
            - si un lllibre dle cas ha estat recomant → +0.5
                - si aquest llibre ha estat valorat (1 o 5) → +0.5
        """

        for i in range(len(user['llibres_recomanats'])): #per cada llibre recomanat
            for k in range(len(users)): #per cada cas similar utilitzat
                #print('son iguals',user['llibres_recomanats'][i], self.cases.iloc[k]['llibres_recomanats'])
                if user['llibres_recomanats'][i] in self.cases.iloc[k]['llibres_recomanats']: #si el llibre recomanat es troba a la llista de llibres recomanats del cas similar
                    self.cases.iloc[k]['utilitat'] += 0.5
                    if user['puntuacions_llibres'][i] == 1 or user['puntuacions_llibres'][i] == 5: #si el llibre recomanat ha rebut una valoracio de 1<x<2 o 4<x<5
                        self.cases.iloc[k]['utilitat'] += 0.5
                        
    def recomana(self, user):
        # user es un diccionari!!!
        users = self.retrieve(user)
        ll = self.reuse(user,users)
        user = self.revise(user, ll)
        user = self.review(user)
        self.retain(user,users)
        if self.iteracions%100==0 and self.iteracions!=0 and self.iteracions!=1:
            self.actualitza_base()
            print('ei nova base amb iteracio\n', self.iteracions)
        #print(self.cases[self.cases.utilitat >0])
        self.iteracions+=1
        return user
    
    def __calculate_optimal_k(self,inertia,k_range):
        
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
    
    def actualitza_base(self):
        casos_utils = self.cases[self.cases.utilitat >0]
        vectors = np.array(self.cases['vector'].tolist())
        for i, row in casos_utils.iterrows():
            vector = row.vector.reshape(1, -1)

            #predim els 3 veins més propers
            nbrs = NearestNeighbors(n_neighbors=3, metric='cosine').fit(vectors)
            _, indexs = nbrs.kneighbors(vector)

            #eliminem els veins que tinguin utilitat 0
            veins_no_utils = self.cases[self.cases.iloc[indexs.flatten()]['utilitat']==0].index
            base_actualitzada = self.cases.drop(veins_no_utils)

        if casos_utils.empty:
            base_actualitzada=self.cases
        
        vectors_actualitzats=list(base_actualitzada.vector)
        wcss = []
        k_range = range(1,11)
        for i in k_range:
            kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42, n_init=10)
            kmeans.fit(vectors_actualitzats)
            wcss.append(kmeans.inertia_)

        kmeans = KMeans(n_clusters=self.__calculate_optimal_k(wcss, k_range))
        base_actualitzada.cluster = kmeans.fit_predict(vectors_actualitzats)
        self.clustering = kmeans
        self.cases = base_actualitzada 
        


            