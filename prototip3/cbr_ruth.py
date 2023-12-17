from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import DistanceMetric
import numpy as np
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
import random
import pandas as pd

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
          
    
    def reuse(self, user,users):
        
        # users és una llista de tuples (usuari, similitud)
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
        _, indices = knn.kneighbors(vector_user)

        # Guardamos los book_ids de los libros más cercanos
        llibres_recom = []
        for i in indices[0]:
            llibres_recom.append(book_ids[i])

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
            llibre_recomanat = self.books[self.books.book_id==int(llibre)].iloc[0]
            # Coger todos los libros que coincidan con el cluster del libro recomendado
            llibres_del_cluster = self.books[self.books['cluster'] == int(cluster.iloc[0])]
            for i,ll in llibres_del_cluster.iterrows():
                if self.similarity(user, ll, "cosine") > self.similarity(user, llibre_recomanat, "cosine"):
                    llibres[llibres.index(llibre)] = ll['book_id']
                    break
        user["llibres_recomanats"] = llibres
        return user
    
    def review(self, user):
        """
        L'usuari valora els tres llibres
        """
        for llibre in user['llibres_recomanats']:
            while True:
                print(f"Quina puntuació li donaries a la recomanació del llibre {self.books.loc[self.books[self.books['book_id'] == int(llibre)].index[0],'title']}? (0-5) ")
                llibre_recomanat = self.books[self.books.book_id==int(llibre)].iloc[0]
                llibres = self.books[self.books.book_id.isin(list(map(int, user['llibres_usuari'])))] #agafem els llibres que s'ha llegit l'usuari
                sim = llibres.apply(lambda x: self.similarity(llibre_recomanat,x,'cosine'),axis=1) #similaritat entre el llibre recomanat i els llibres llegits per l'usuari
                #llibres.loc[sim.nlargest(3).index] #agafo els índexs dels llibres més similars
                #list(llibres.loc[sim.nlargest(3).index]['book_id']) books id dels llibres per tornar-los a agafar dels recomenats
                books_id_similars = list(llibres.loc[sim.nlargest(3).index]['book_id'])
                indexos = [i for i, el in enumerate(user['llibres_usuari']) if int(el) in books_id_similars]  #agafo indexs dins la llista dels llibres per accedir a la valoracio
                puntuacio = round(np.mean([user['val_llibres'][i] for i in indexos])) #mitjana de les valoracions dels llibres similars
                puntuacio = puntuacio if puntuacio != 0 else 1
                print(books_id_similars)
                print(indexos)
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
            print(self.cases[self.cases.utilitat >0])
            self.actualitza_base()
            print('ei nova base amb iteracio\n', self.iteracions)
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
            print('no hi ha nova!!!')
            base_actualitzada=self.cases
        else:
            print('hi ha nova!!!')
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
        


            