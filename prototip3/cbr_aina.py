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
        
        distancies = veins['vector'].apply(lambda x: np.linalg.norm(vector - np.array(list(x)), axis=1)) #distancia euclidea 
        #distancies = veins['vector'].apply(lambda x: self.similarity(user,x,'cosine'))
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
    
    def retain(self, user, users, ll):
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
        
        self.utilitat(user,ll,users) # actualitzem utilitat

    def utilitat(self, user, llibres, casos):
        """
        user = usuari final
        llibres = llibres del reuse
        casos = casos del retrieve
        ---------------------
        Calcula la utilitat de l'usuari
            - calculem utilitat al retain
            - cas utilitzat → llibre del cas és recomanat → llibre recomanat bona valoració → ENTENEM QUE EL CAS ÉS ÚTIL
            - utilitat progressiva:
                - si llibre del cas passa reuse → +0.5
                - si llibre del cas passa revise → +0.5
                - si llibre del cas passa review → +0.5
        """
        comptador = 0
        for llibre in llibres: #si el llibre ha passat la fase reuse
            for k in range(len(casos)):
                if llibre in self.cases.iloc[k]['llibres_recomanats']:
                    self.cases.iloc[k]['utilitat'] += 0.5
                    if llibre in user['llibres_recomanats']: #si el llibre ha passat la fase revise
                        self.cases.iloc[k]['utilitat'] += 0.5
                        if user['puntuacions_llibres'][comptador] == 1 or user['puntuacions_llibres'][comptador] == 5: #si el llibre recomanat ha rebut una valoracio de 1<x<2 o 4<x<5
                            self.cases.iloc[k]['utilitat'] += 0.5
            comptador += 1
    
    def justifica(self, user, users, llibres):
        """
        user = usuari final
        llibres = llibres del reuse
        users = casos del retrieve
        ---------------------
        Justifica per que li recomanem un llibre a l'usuari:
        - si el llibre procedeix directament d'un dels casos del retrive (és a dir, el llibre és de l'output del reuse i del revise) 
            --> Justificacio: 'perquè hi ha lectors com tu que els hi agrada!'
        - si el llibre procedeix del revise, és a dir, no és directament d'un cas del retrive 
            --> Justificacio: 'perquè és un llibre que et podria agradar ja que té aquestes 3 caracteristiques!'
            - per saber quines caracteristiques volem destacar agafem tots els users de la base de dades de casos 
                que hagin llegit aquest llibre, fem la mitjana entre els seus vectors i comparem amb el nostre 
                vector usuari: less tres components més semblants seran les tres caracteristiques que destacarem
        - si el llibre procedeix de la pregunta de quin tipus de recomanació vol el user
            --> Justificació: 'perque vols una recomanació de x tipus1'
        """
        casos = users #casos retrieve
        ll = llibres #llibres el reuse

        for llibre in user['llibres_recomanats']:
            justificacio = []
            justificacio.append(f'Et recomanem el llibre {self.books.loc[self.books[self.books["book_id"] == int(llibre)].index[0],"title"]}')
            
            #si el llibre pertany a un dels llibres recomanats del reuse
            if llibre in ll:
                for i in range(len(casos)): #mirem casos el retrieve
                    if llibre in casos[i][0]['llibres_recomanats']: #mirar si el llibre es troba a la llista de llibres recomanats del cas similar
                        justificacio.append(' perquè hi ha lectors com tu que els hi agrada!')
                        break
                print(justificacio)
                pass
            
            #si el llibre correspon a la pregunta de "quin tipus de recomanació vols?"
            elif llibre == user['llibres_recomanats'][3]:
                justificacio.append(f' perquè vols una recomanació {user["tipus_recomanacio"]}')
                print(justificacio)
                pass

            #si el llibre pertany a un dels llibres recomanats del revise
            else:
                #agafem tots els users de la base de dades de casos que hagin llegit aquest llibre
                users_llibre = self.cases[self.cases['llibres_recomanats'].apply(lambda x: llibre in x)]
                #fem la mitjana entre els seus vectors
                vector_mitja = np.mean(np.array(users_llibre['vector'].tolist()), axis=0)
                #comparem amb el nostre vector usuari: less tres components més semblants seran les tres caracteristiques que destacarem
                vector_user = user['vector']
                similaritat = self.similarity(vector_user, vector_mitja, 'cosine')
                indexs = np.argsort(similaritat)[:3]
                caracteristiques = np.array(self.books.columns)[indexs]
                justificacio.append(f" perquè té aquestes 3 caracteristiques que t'agraden: {caracteristiques[0]}, {caracteristiques[1]} i {caracteristiques[2]}")
                print(justificacio)
                pass

    def recomana(self, user):
        # user es un diccionari!!!
        users = self.retrieve(user)
        ll = self.reuse(user,users)
        user = self.revise(user, ll)
        user = self.review(user)
        self.retain(user,ll,users)
        if self.iteracions%100==0 and self.iteracions!=0 and self.iteracions!=1:
            self.actualitza_base()
            print('ei nova base amb iteracio\n', self.iteracions)
        #print(self.cases[self.cases.utilitat >0])
        self.iteracions+=1
        self.justifica(user, users, ll)
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
        self.cases = base_actualitzada 
        


            