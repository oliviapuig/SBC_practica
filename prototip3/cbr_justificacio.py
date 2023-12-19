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
    
    def reuse(self, user, users):
        
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
                    if llibre not in user['llibres_usuari']:
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
        #user["llibres_recomanats"] = llibres
        for llibre in llibres: # Bucle executat 3 vegades
            cluster = self.books[self.books.book_id==int(llibre)]["cluster"].iloc[0]
            llibre_recomanat = self.books[self.books.book_id==int(llibre)].iloc[0]  # Tot el llibre recomanat
            llibres_del_cluster = self.books[self.books['cluster'] == int(cluster)] # Tots els llibres del cluster
            for i, ll in llibres_del_cluster.iterrows():
                if ll['book_id'] not in llibres and ll['book_id'] not in user["llibres_usuari"]: # Si no esta recomanat i no l'ha llegit l'usuari
                    if len(llibres) == 3:
                        if self.similarity(user, ll, "cosine") > self.similarity(user, llibre_recomanat, "cosine"):
                            print('he entrat')
                            llibres[llibres.index(llibre)] = ll['book_id']  # Canviem el llibre recomanat per un altre del cluster
                            llibre_recomanat = ll
                            break
                    else:
                        # Recalcula i agafa el següent llibre més probable
                        user["llibres_recomanats"].append(ll['book_id'])
                        pass

        user["llibres_recomanats"] = llibres[:2]

        print("A continuació, et demanarem algunes preferències per millorar la recomanació.")

        continuar = True
        while continuar:

            # Preguntas de preferencias al usuario
            #preferencia_llibre = input("Prefereixes llibres semblants als llegits o vols explorar? (Semblants/Explorar): ").lower()
            #preferencia_popularitat = input("Prefereixes llibres populars (bestseller) o no tan populars? (Bestseller/No tan popular): ").lower()
            preferencia_llibre = np.random.choice(['semblants', 'explorar'])
            preferencia_popularitat = np.random.choice(['bestseller', 'no tan popular'])

            # Lògica per ajustar les recomanacions segons les preferències del usuari
            if preferencia_llibre == 'semblants':
                if preferencia_popularitat == 'bestseller':
                    llibres_del_cluster = llibres_del_cluster[llibres_del_cluster['bestseller'] == True]
                    distancies = llibres_del_cluster.apply(lambda x: self.similarity(user, x, 'cosine'), axis=1)
                    index_llibre_mes_semblant = distancies.idxmax()
                    user["llibres_recomanats"].append(llibres_del_cluster.loc[index_llibre_mes_semblant, 'book_id'])
                    continuar = False
                
                elif preferencia_popularitat == 'no tan popular':
                    llibres_del_cluster = llibres_del_cluster[llibres_del_cluster['bestseller'] == False]
                    distancies = llibres_del_cluster.apply(lambda x: self.similarity(user, x, 'cosine'), axis=1)
                    index_llibre_mes_semblant = distancies.idxmax()
                    user["llibres_recomanats"].append(llibres_del_cluster.loc[index_llibre_mes_semblant, 'book_id'])
                    continuar = False
                    
                else:
                    print("Opció no vàlida per preferencia_popularitat. Si us plau, respon 'bestseller' o 'no tan popular'.")

            elif preferencia_llibre == 'explorar':
                distancies_cluster = self.clustering.transform(user.vector.reshape(1, -1))
                cluster_mes_proper = distancies_cluster.argsort()[0][1]
                
                # Filtra els llibres del cluster més proper
                llibres_cluster_mes_proper = self.books[self.books['cluster'] == int(cluster_mes_proper)]

                # Filtra pels llibres que són bestsellers o no, segons la preferència de l'usuari
                if preferencia_popularitat == 'bestseller':
                    llibres_recomanats_cluster_mes_proper = llibres_cluster_mes_proper[llibres_cluster_mes_proper['bestseller'] == True]
                elif preferencia_popularitat == 'no tan popular':
                    llibres_recomanats_cluster_mes_proper = llibres_cluster_mes_proper[llibres_cluster_mes_proper['bestseller'] == False]
                else:
                    print("Opció no vàlida per preferencia_popularitat. Si us plau, respon 'bestseller' o 'no tan popular'.")
                    return user
                
                # Afegeix un llibre aleatori de la llista filtrada als llibres recomanats de l'usuari
                user["llibres_recomanats"].append(llibres_recomanats_cluster_mes_proper.sample(1)['book_id'].values[0])
                continuar = False
            
            else:
                print("Opció no vàlida per preferencia_llibre. Si us plau, respon 'semblants' o 'explorar'.")

        # posar en una llista els motius de la recomanació
        user['motius_recomanacio'] = []
        user['motius_recomanacio'].append(preferencia_llibre)
        user['motius_recomanacio'].append(preferencia_popularitat)
        user['motius_recomanacio'].append(cluster)
        
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
                #print(books_id_similars)
                #print(indexos)
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
        
        self.utilitat(user, ll, users) # actualitzem utilitat
        print('llista retain', ll)

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

    def justifica(self, user, users, llibres, llibres_recom):
        casos = users #casos retrieve
        ll = self.reuse(user, users) #llibres reuse
        print('soc dins de la funció')
        for llibre in user['llibres_recomanats']:
            justificacio = []
            justificacio.append(f'Et recomanem el llibre {self.books.loc[self.books[self.books["book_id"] == int(llibre)].index[0],"title"]}')
            print('soc dins del for')
            # comprovar si el llibre de l'output del reuse i del revise
            if llibre in llibres_recom:
                justificacio.append('perquè hi ha lectors com tu que els hi agrada!')
                print('soc dins del if')
            # elif justificant quan el llibre procedeix del chatbot del revise user[motius_recomanacio]
                if llibre in user['llibres_recomanats']:
                    print('soc dins del elif')
                    justificacio.append(f"perquè vols una recomanació de {user['motius_recomanacio'][0]} i {user['motius_recomanacio'][1]}")
                    # si pertany al cluster del user
                    if self.books[self.books.book_id==int(llibre)]['cluster'].iloc[0] == user['motius_recomanacio'][2]:
                        justificacio.append('perquè és un llibre del mateix cluster que el teu')
                    else:
                        justificacio.append('perquè és un llibre d\'un cluster semblant al teu, però un xic més arriscat')
                else:
                    print('soc dins del else')
                    # si el llibre no és del output del reuse ni del revise
                    # agafem tots els users de la base de dades de casos que hagin llegit aquest llibre
                    # fem la mitjana entre els seus vectors i comparem amb el nostre vector usuari
                    # les tres components més semblants seran les tres caracteristiques que destacarem
                    users_llibre = self.cases[self.cases['llibres_recomanats'].apply(lambda x: llibre in x)]
                    vectors = np.array(users_llibre['vector'].tolist())
                    vector_usuari = user['vector'].reshape(1, -1)
                    distancies = []
                    for i in range(len(vectors)):
                        distancies.append(self.similarity(user, users_llibre.iloc[i], 'cosine'))
                    distancies = np.array(distancies)
                    indexs = distancies.argsort()[-3:][::-1]
                    caracteristiques = []
                    for i in indexs:
                        caracteristiques.append(self.books[self.books.book_id==int(llibre)].iloc[0].index[np.argmax(users_llibre.iloc[i]['vector'])])
                    justificacio.append(f"perquè és un llibre que et podria agradar ja que té aquestes 3 caracteristiques: {caracteristiques[0]}, {caracteristiques[1]} i {caracteristiques[2]}")
                
                # imprimeix justificacio
                print(justificacio)

                        
    def recomana(self, user):
        # user es un diccionari!!!
        users = self.retrieve(user)
        ll = self.reuse(user, users)
        user = self.revise(user, ll)
        user = self.review(user)
        self.retain(user, users, ll)
        if self.iteracions%100==0 and self.iteracions!=0 and self.iteracions!=1:
            self.actualitza_base()
            print('ei nova base amb iteracio\n', self.iteracions)
        #print(self.cases[self.cases.utilitat >0])
        self.iteracions+=1
        self.justifica(user, users, ll, llibres_recom)
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


            