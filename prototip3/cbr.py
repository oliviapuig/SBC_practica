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
        self.books = books
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
                    print('he entrat')
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
        
    def __scale(self, vector, min_ant = 0, max_ant = 5, min_nou = 0, max_nou = 1):
        """
        Passar de una valoracio [0-5] a una puntuació [-1-1]
        """
        from sklearn.preprocessing import MinMaxScaler
        if isinstance(vector, int):
            vector = np.array([vector])
        if vector.shape[0] > 1:
            min_ant = min(vector)
            #max_ant = max(vector)
        escalador = MinMaxScaler(feature_range=(min_nou, max_nou))
        escalador.fit([[min_ant], [max_ant]])
        return escalador.transform(vector.reshape(-1, 1)).flatten()

    def __get_attributes(self, llibres_usuari, val_llibres):
        """
        Aconseguir el vector d'atributs d'usuari a partir dels llibres que ha llegit
        """
        len_vector = len(self.books["vector"].iloc[0])
        vector_usuari = np.zeros(len_vector)
        for ll, val in zip(llibres_usuari, val_llibres):
            vector_usuari += np.array(self.books[self.books["book_id"] == int(ll)]["vector"].iloc[0]) * self.__scale(val)
        np.round(vector_usuari, 1)

        vector_usuari = self.__scale(vector_usuari, min(vector_usuari), max(vector_usuari), 0, 1)

        # Si hay vectores con valores entre -0.01 y 0.01, los ponemos a 0
        for i in range(len(vector_usuari)):
            if vector_usuari[i] < 0.01 and vector_usuari[i] > -0.01:
                vector_usuari[i] = 0

        return np.round(vector_usuari, 4)

    def inicia(self, fitxer):
        """
        Agafa el fitxer, l'obra i posa les dades del cas a una nova base de dades.
        """
        usuaris = {}
        with open(fitxer, 'rb') as arxiu:
            for linia in arxiu:
                # Split de la linia por espacios
                linia = linia.decode("utf-8").split()
                # La linia[0] es el id del usuario
                if linia[0] not in usuaris:
                    usuaris[linia[0]] = {"llibres_usuari": [], "val_llibres": []}
                # La linia[1] es el id del libro
                # Aseguramos que el libro existe
                assert int(linia[1]) in self.books.book_id.values, f"El llibre {linia[1]} no existeix"
                # Aseguramos que el libro no haya estado valorado por el usuario
                assert int(linia[1]) not in usuaris[linia[0]]["llibres_usuari"], f"El llibre {linia[1]} ja ha estat valorat per l'usuari {linia[0]}"
                usuaris[linia[0]]["llibres_usuari"].append(int(linia[1]))
                # La linia[2] es la valoracion del libro
                usuaris[linia[0]]["val_llibres"].append(int(linia[2]))
        
        for user in usuaris:
            vector_usuari = self.__get_attributes(usuaris[user]["llibres_usuari"], usuaris[user]["val_llibres"])
            usuaris[user]["vector"] = vector_usuari
            # Predict del cluster
            usuaris[user]["cluster"] = self.clustering.predict(vector_usuari.reshape(1,-1))[0]
            # Añadimos los libros recomendados
            usuaris[user]["llibres_recomanats"] = []
            usuaris[user]["puntuacions_llibres"] = []
            usuaris[user]["utilitat"] = 0
        # Añadimos los usuarios a una nueva base de datos
        db_nou = pd.DataFrame(usuaris).T
        # Afegir columna del user_id
        db_nou["user_id"] = db_nou.index
        db_nou.reset_index(drop=True, inplace=True)
        # Reordenar las columnas
        db_nou = db_nou[["user_id", "llibres_usuari", "val_llibres", "llibres_recomanats", "puntuacions_llibres", "cluster", "utilitat", "vector"]]
        for i in range(len(db_nou)):
            db_nou.at[i, 'vector'] = np.array(db_nou.at[i, 'vector'])

        return db_nou


            