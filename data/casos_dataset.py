json_name = 'data.json'
pkl_name = 'casos.pkl'
csv_name = 'casos.csv'
carpeta = 'data/'
pkl_name_ll = 'llibres.pkl'
csv_name_ll = 'llibres.csv'
path_reviews_dataset = 'https://datarepo.eng.ucsd.edu/mcauley_group/gdrive/goodreads/goodreads_reviews_dedup.json.gz'
path_fitxer_llibres = '/Users/ucemarc/Downloads/goodreads_books.json'
path_genres_dataset = '/Users/ucemarc/Downloads/goodreads_book_genres_initial.json'

import requests
import gzip
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import MinMaxScaler
np.random.seed(0)

# if casos.pkl exists, load it
try:
    casos = pd.read_pickle(carpeta+pkl_name)
    get = False
    print("casos.pkl exists. Loading...")
except:
    print("Starting to create casos.pkl")
    get = True

# Make a dataset with users that have rated more than 10 books and less than 50
if get:
    # Descarregar el dataset
    url = path_reviews_dataset

    # Realizar la solicitut HTTP GET
    response = requests.get(url, stream=True)

    # Verificar si la solicitud fue exitosa (código de estado 200)
    if response.status_code == 200:
        # Descomprimir el contenido del archivo
        with gzip.GzipFile(fileobj=response.raw) as f:
            # Leer las primeras 500 filas del JSON
            primeras_500_filas = [json.loads(next(f)[:-1].decode('utf-8')) for _ in range(500000)]

        print("JSON creat.")
    else:
        print(f"Error al descargar el archivo. Código de estado: {response.status_code}")

    # Read eoo.json only user_id, book_id, rating
    df = pd.DataFrame(primeras_500_filas)
    df = df[['user_id', 'book_id', 'rating']]

    # Plot rating distribution and save to eoo/rating_distribution.png
    sns.set_style('darkgrid')
    plt.figure(figsize=(10, 6))
    sns.countplot(x='rating', data=df)
    plt.xlabel('Rating')
    plt.ylabel('Count')
    plt.title('Rating Distribution')
    plt.savefig(f'{carpeta}rating_distribution.png')

    # Give me unique users
    unique_users = df['user_id'].unique()

    # Make a database with unique users, list of books rated and list of rating for each book
    df_aux = pd.DataFrame(columns=['user_id', 'books', 'ratings'])

    for user in unique_users:
        # Filter by user
        user_df = df[df['user_id'] == user]
        # Get list of books rated by user
        books = user_df['book_id'].tolist()
        # Get list of ratings for each book
        ratings = user_df['rating'].tolist()
        # Create a dictionary with books and ratings
        user_dict = dict(zip(books, ratings))
        # Save user, books and ratings in df_aux using pd.concat
        df_aux = pd.concat([df_aux, pd.DataFrame({'user_id': [user], 'books': [books], 'ratings': [ratings]})])

    df_aux = df_aux.reset_index(drop=True)

    print("Dataset joined. Unique users:", len(df_aux))

    # Plot how many books each user has rated and save to eoo/books_rated_before.png
    plt.figure(figsize=(10, 6))
    plt.xlabel('user_id')
    plt.ylabel('Number of books rated')
    plt.title('Number of books rated by each user')
    plt.plot(df_aux['user_id'], df_aux['books'].apply(lambda x: len(x)))
    plt.savefig(f'{carpeta}books_rated_before.png')

    min_books = 10
    max_books = 20

    # Remove users that have rated less than 10 books and more than 50
    df_aux = df_aux[df_aux['books'].apply(lambda x: len(x) >= min_books and len(x) <= max_books)]
    df_aux = df_aux.reset_index(drop=True)

    print(f"Dataset filtered with users with more than {min_books} and less than {max_books} books reviewed. Unique users:", len(df_aux))

    # Plot how many books each user has rated and save to eoo/books_rated_after.png
    plt.figure(figsize=(10, 6))
    plt.xlabel('user_id')
    plt.ylabel('Number of books rated')
    plt.title('Number of books rated by each user')
    plt.plot(df_aux['user_id'], df_aux['books'].apply(lambda x: len(x)))
    plt.savefig(f'{carpeta}books_rated_after.png')

    # For each user get 3 last books and their ratings and put them in a new column "llibres_recomanats" i "puntuacions_llibres". Then remove the 3 books from the list of books rated by the user.
    df_aux['llibres_recomanats'] = df_aux['books'].apply(lambda x: x[-3:])
    df_aux['puntuacions_llibres'] = df_aux['ratings'].apply(lambda x: x[-3:])
    df_aux['books'] = df_aux['books'].apply(lambda x: x[:-3])
    df_aux['ratings'] = df_aux['ratings'].apply(lambda x: x[:-3])

    print("Done creating new columns.")

    # Change "books" and "ratings" columns to "llibres_usuari" and "val_llibres"
    df_aux = df_aux.rename(columns={'books': 'llibres_usuari', 'ratings': 'val_llibres'})

    df_aux.to_pickle(pkl_name)
    df_aux.to_csv(csv_name, index=False)

casos = pd.read_pickle(carpeta+pkl_name)

# If llibres.pkl exists, load it
try:
    llibres = pd.read_pickle(carpeta+pkl_name_ll)
    get = False
    print("llibres.pkl exists. Loading...")
except:
    print("Starting to create llibres.pkl")
    get = True

if get:
    # For each row, add all the books from "llibres_usuari" and "llibres_recomanats" to a set
    set_llibres = set()
    for index, row in casos.iterrows():
        for llibre in row['llibres_usuari']:
            set_llibres.add(llibre)
        for llibre in row['llibres_recomanats']:
            set_llibres.add(llibre)

    set_llibres = list(set_llibres)

    fitxer = path_fitxer_llibres
    # Crear un DataFrame vacío para almacenar los libros que coincidan
    df_llibres = pd.DataFrame(columns=['isbn', 'book_id', 'similar_books', 'average_rating', 'description', 'authors', 'isbn13', 'num_pages', 'publication_year', 'title', 'language_code', 'format', 'series'])

    # Leer el archivo línea por línea
    i = 1
    with open(fitxer, 'r', encoding='utf-8') as file:
        for line in file:
            book = json.loads(line)
            if book['book_id'] in set_llibres:
                # Only keep the columns "isbn", "book_id", "similar_books", "average_rating", "similar_books", "description", "authors", "isbn13", "num_pages", "publication_year", "title" and "language_code"
                book = {k: book[k] for k in ['isbn', 'book_id', 'similar_books', 'average_rating', 'similar_books', 'description', 'authors', 'isbn13', 'num_pages', 'publication_year', 'title', 'language_code', 'format', 'series']}
                aut = []
                for author in book['authors']:
                    aut.append(author['author_id'])
                book['authors'] = aut
                # Convert the dictionary to a DataFrame
                book = pd.DataFrame([book], index=[0])
                # Add the book to the DataFrame
                df_llibres = pd.concat([df_llibres, pd.DataFrame(book, index=[0])])
                i += 1
    print("Done creating llibres.pkl")
    df_llibres.to_csv(carpeta+csv_name_ll, index=False)
    df_llibres.to_pickle(carpeta+pkl_name_ll)

# If column "genres" exists in llibres.pkl then get = False
try:
    llibres = pd.read_pickle(carpeta+pkl_name_ll)
    llibres['genres']
    get = False
    print("Column genre already created. Loading...")
except:
    print("Starting to create column 'genres' in llibres.pkl")
    get = True
    df_llibres = pd.read_csv(carpeta+csv_name_ll)

if get:
    fitxer = path_genres_dataset

    # Crear un DataFrame vacío para almacenar los libros que coincidan
    df_genres = pd.DataFrame(columns=['book_id', 'genres'])

    with open(fitxer, 'r', encoding='utf-8') as file:
        for line in file:
            book = json.loads(line)
            if book['book_id'] in set_llibres:
                # Only keep the columns "isbn", "book_id", "similar_books", "average_rating", "similar_books", "description", "authors", "isbn13", "num_pages", "publication_year", "title" and "language_code"
                book = {k: book[k] for k in ['book_id', 'genres']}
                # Get only the keys of the dictionary
                book['genres'] = list(book['genres'].keys())
                # Convert the dictionary to a DataFrame
                book = pd.DataFrame([book], index=[0])
                # Add the book to the DataFrame
                df_genres = pd.concat([df_genres, pd.DataFrame(book, index=[0])])
                #df_genres.to_csv("genres.csv", index=False)

    # Merge df_llibres and df_genres on book_id
    df_llibres['book_id'] = df_llibres['book_id'].astype(int)
    df_genres['book_id'] = df_genres['book_id'].astype(int)
    df_llibres= pd.merge(df_llibres, df_genres, on='book_id', how='inner')
    df_llibres.to_csv("llibres.csv", index=False)

    # Check how many unique genres there are
    unique_genres = set()
    for index, row in df_llibres.iterrows():
        for genre in row['genres']:
            unique_genres.add(genre)

    # Replace 'history, historical fiction, biography' to 'history'
    df_llibres['genres'] = df_llibres['genres'].apply(lambda x: ['history' if i == 'history, historical fiction, biography' else i for i in x])
    # Replace 'fantasy, paranormal' to 'fantasy'
    df_llibres['genres'] = df_llibres['genres'].apply(lambda x: ['fantasy' if i == 'fantasy, paranormal' else i for i in x])
    # Replace 'mystery, thriller, crime' to 'mystery'
    df_llibres['genres'] = df_llibres['genres'].apply(lambda x: ['mystery' if i == 'mystery, thriller, crime' else i for i in x])
    # Replace 'comics, graphic' to 'comics'
    df_llibres['genres'] = df_llibres['genres'].apply(lambda x: ['comics' if i == 'comics, graphic' else i for i in x])
    df_llibres.to_csv(carpeta+csv_name_ll, index=False)
    df_llibres.to_pickle(carpeta+pkl_name_ll)

    # Check how many unique genres there are
    unique_genres = set()
    for index, row in df_llibres.iterrows():
        for genre in row['genres']:
            unique_genres.add(genre)
    print("Done creating column 'genres' in llibres.pkl")

llibres = pd.read_pickle(carpeta+pkl_name_ll)
casos = pd.read_pickle(carpeta+pkl_name)

# If column "estil_literari" exists in llibres.pkl then get = False
try:
    llibres['estil_literari']
    get = False
    print("llibres.pkl already preprocessed. Loading...")
except:
    print("Starting to preprocess llibres.pkl")
    get = True

if get:
    categories = {
    "estil_literari": ["realisme", "romanticisme", "naturalisme", "simbolisme", "modernisme", "realisme magico", "postmodernisme"],
    "complexitat": ["baixa", "mitjana", "alta"],
    "caracteristiques": ["simples", "complexes"],
    "desenvolupament_del_personatge": ["baix", "mitja", "alt"],
    "accio_o_reflexio": ["accio", "reflexio"],
    "epoca": ["actual", "passada", "futura"],
    "detall_cientific": ["baix", "mitja", "alta"]
    }
    
    def make_vector(length1, length2, unique_min, unique_max, categorie):
        # Número de valores únicos (entre 2 y 4)
        num_unique_values = np.random.randint(unique_min, unique_max)

        # Seleccionar valores únicos de forma aleatoria
        unique_values = np.random.choice(categories[categorie], size=num_unique_values, replace=False)

        # Crear el vector de 10 posiciones
        vector1 = [np.random.choice(unique_values) for _ in range(length1)]
        vector2 = [np.random.choice(unique_values) for _ in range(length2)]
        return vector1, vector2

    # Funció auxiliar per actualitzar els diccionaris
    def actualitzar_diccionaris(llibre_id, valor, diccionari):
        if valor in diccionari[llibre_id]:
            diccionari[llibre_id][valor] += 1
        else:
            diccionari[llibre_id][valor] = 1

    # Inicialització de diccionaris per a cada atribut
    estil_literari = [{} for _ in range(len(llibres))]
    complexitat = [{} for _ in range(len(llibres))]
    caracteristiques = [{} for _ in range(len(llibres))]
    desenvolupament_del_personatge = [{} for _ in range(len(llibres))]
    accio_o_reflexio = [{} for _ in range(len(llibres))]
    epoca = [{} for _ in range(len(llibres))]
    detall_cientific = [{} for _ in range(len(llibres))]

    for index, row in casos.iterrows():
        len_llibres_usuari = len(row['llibres_usuari'])
        len_llibres_recomanats = len(row['llibres_recomanats'])
        estil_literari1, estil_literari2 = make_vector(len_llibres_usuari, len_llibres_recomanats, 2, 4, "estil_literari")
        complexitat1, complexitat2 = make_vector(len_llibres_usuari, len_llibres_recomanats, 1, 3, "complexitat")
        caracteristiques1, caracteristiques2 = make_vector(len_llibres_usuari, len_llibres_recomanats, 1, 3, "caracteristiques")
        desenvolupament_del_personatge1, desenvolupament_del_personatge2 = make_vector(len_llibres_usuari, len_llibres_recomanats, 1, 3, "desenvolupament_del_personatge")
        accio_o_reflexio1, accio_o_reflexio2 = make_vector(len_llibres_usuari, len_llibres_recomanats, 1, 2, "accio_o_reflexio")
        epoca1, epoca2 = make_vector(len_llibres_usuari, len_llibres_recomanats, 1, 3, "epoca")
        detall_cientific1, detall_cientific2 = make_vector(len_llibres_usuari, len_llibres_recomanats, 1, 3, "detall_cientific")

        for i in range(len_llibres_usuari):
            llibre_id_usuari = llibres[llibres["book_id"] == int(row['llibres_usuari'][i])].index[0]
            actualitzar_diccionaris(llibre_id_usuari, estil_literari1[i], estil_literari)
            actualitzar_diccionaris(llibre_id_usuari, complexitat1[i], complexitat)
            actualitzar_diccionaris(llibre_id_usuari, caracteristiques1[i], caracteristiques)
            actualitzar_diccionaris(llibre_id_usuari, desenvolupament_del_personatge1[i], desenvolupament_del_personatge)
            actualitzar_diccionaris(llibre_id_usuari, accio_o_reflexio1[i], accio_o_reflexio)
            actualitzar_diccionaris(llibre_id_usuari, epoca1[i], epoca)
            actualitzar_diccionaris(llibre_id_usuari, detall_cientific1[i], detall_cientific)

        for i in range(len_llibres_recomanats):
            llibre_id_recomanat = llibres[llibres["book_id"] == int(row['llibres_recomanats'][i])].index[0]
            actualitzar_diccionaris(llibre_id_recomanat, estil_literari2[i], estil_literari)
            actualitzar_diccionaris(llibre_id_recomanat, complexitat2[i], complexitat)
            actualitzar_diccionaris(llibre_id_recomanat, caracteristiques2[i], caracteristiques)
            actualitzar_diccionaris(llibre_id_recomanat, desenvolupament_del_personatge2[i], desenvolupament_del_personatge)
            actualitzar_diccionaris(llibre_id_recomanat, accio_o_reflexio2[i], accio_o_reflexio)
            actualitzar_diccionaris(llibre_id_recomanat, epoca2[i], epoca)
            actualitzar_diccionaris(llibre_id_recomanat, detall_cientific2[i], detall_cientific)
    
    # Choose the most voted value for each book
    for i in range(len(llibres)):
        if len(estil_literari[i]) > 0:
            estil_literari[i] = max(estil_literari[i], key=estil_literari[i].get)
        if len(complexitat[i]) > 0:
            complexitat[i] = max(complexitat[i], key=complexitat[i].get)
        if len(caracteristiques[i]) > 0:
            caracteristiques[i] = max(caracteristiques[i], key=caracteristiques[i].get)
        if len(desenvolupament_del_personatge[i]) > 0:
            desenvolupament_del_personatge[i] = max(desenvolupament_del_personatge[i], key=desenvolupament_del_personatge[i].get)
        if len(accio_o_reflexio[i]) > 0:
            accio_o_reflexio[i] = max(accio_o_reflexio[i], key=accio_o_reflexio[i].get)
        if len(epoca[i]) > 0:
            epoca[i] = max(epoca[i], key=epoca[i].get)
        if len(detall_cientific[i]) > 0:
            detall_cientific[i] = max(detall_cientific[i], key=detall_cientific[i].get)
    
    # Afegir les noves columnes al DataFrame
    llibres["estil_literari"] = estil_literari
    llibres["complexitat"] = complexitat
    llibres["caracteristiques"] = caracteristiques
    llibres["desenvolupament_del_personatge"] = desenvolupament_del_personatge
    llibres["accio_o_reflexio"] = accio_o_reflexio
    llibres["epoca"] = epoca
    llibres["detall_cientific"] = detall_cientific

    print("Added new columns to llibres.pkl")
    llibres.to_pickle(carpeta+pkl_name_ll)
    llibres.to_csv(carpeta+csv_name_ll, index=False)

    # FUNCIÓ PER ELIMINAR DE SIMILARS AQUELLS LLIBRES QUE NO ESTAN A LA BASE DE DADES

    # Funció per convertir la cadena de la llista en una llista real i netejar-la
    def neteja_similars(similars, ids_valids):
        # Convertir la cadena a una llista
        similars_list = similars.strip("[]").replace("'", "").split(", ")
        # Mantenir només els IDs que estan presents en ids_valids
        return [id for id in similars_list if id in ids_valids]

    # Obtenir els book_id com a conjunt per a una cerca més ràpida
    ids_valids = set(df['book_id'].astype(str))
    llibres['similar_books'] = llibres['similar_books'].apply(lambda x: neteja_similars(x, ids_valids))
    print("Removing similar books that are not in the database")

    # Funció per assignar 'noisbn'
    def assigna_noisbn(valor):
        if pd.isna(valor) or valor == 'NaN':
            return 'noisbn'
        else:
            return valor

    # Aplicar la funció a les columnes isbn i isbn13
    llibres['isbn'] = llibres['isbn'].apply(assigna_noisbn)
    llibres['isbn13'] = llibres['isbn13'].apply(assigna_noisbn)
    print("Done preprocessing isbn and isbn13")

    df = pd.read_pickle(carpeta+pkl_name)
    # Unificar les diferents categories de audio
    df['format'] = df['format'].apply(lambda x: 'audio' if x in ['Audible Audio', 'Audio CD', 'Audio Cassette', 'Audio', 'Audiobook', 'audio cd', 'MP3 CD'] else x)
    # Unificar les diferents categories de ebook
    df['format'] = df['format'].apply(lambda x: 'ebook' if x in ['ebook', 'Kindle Edition', 'HTML', 'Kindle', 'chapbook/ebook', 'Serialized Digital Download'] else x)
    # Unificar les diferents categories de paper
    df['format'] = df['format'].apply(lambda x: 'tapa blanda' if x in ['Paperback', 'Mass Market Paperback', 'paper', 'Trade Paperback', 'pocket', 'Softcover', 'Trade paperback', 'Paper Back', 'Perfect Paperback', 'paperback', 'Trade Paper', 'Paberback', 'Softcover with Flap', 'Tapa blanda con solapas'] else x)
    # Unificar les diferents categories de hardcover
    df['format'] = df['format'].apply(lambda x: 'tapa dura' if x in ['Hardcover', 'Board book', 'Board Book', 'Hardback', 'hardcover', 'issue', 'Broche', 'Klappenbroschur', 'Nook', 'Library Binding', 'Gebunden', 'Wen Ku', 'Leather Bound', 'Musc. Supplies', 'Podiobook', 'Brossura', 'Nook Book', 'Spiral-bound', 'Novelty Book', 'Glf `dy,', 'Misc. Supplies', 'Broschiert', 'Unknown Binding'] else x)
    # Unificar les diferents categories de comic
    df['format'] = df['format'].apply(lambda x: 'tapa blanda' if x in ['comics', 'Thirteen interactive chapters.', 'Graphic Novel', 'Digital comic', 'Comic Book'] else x)
    # Poner los nan a las otras categorias siguiendo la distribución de las categorias actuales
    df['format'] = df['format'].apply(lambda x: np.random.choice(['tapa blanda', 'tapa dura', 'ebook']) if pd.isna(x) else x)

    # Cambiamos los valores menores a 1.5 en average_rating a la media
    df['average_rating'] = df['average_rating'].apply(lambda x: np.random.uniform(1.5, 5) if x < 1.5 else x)
    print("Done preprocessing format and average_rating")

    # Rellenar num_pages y publication_year
    import requests

    def obtener_info_libro(isbn):
        base_url = "https://openlibrary.org/api/books"
        params = {
            "bibkeys": f"ISBN:{isbn}",
            "format": "json",
            "jscmd": "data",
        }

        try:
            response = requests.get(base_url, params=params)
            data = response.json()

            if f"ISBN:{isbn}" in data:
                book_info = data[f"ISBN:{isbn}"]
                return book_info  # Devuelve todos los campos disponibles
            else:
                return "No se encontró información para ese ISBN."
        except Exception as e:
            return f"Error: {str(e)}"

    # Para los libros que no tienen num_pages, obtener el número de páginas de OpenLibrary
    for index, row in df.iterrows():
        if pd.isna(row['num_pages']) or pd.isna(row['publication_year']):
            isbn = row['isbn13']
            if isbn != 'noisbn':
                info_libro = obtener_info_libro(isbn)
                if info_libro != "No se encontró información para ese ISBN.":
                    if pd.isna(row['num_pages']):
                        try:
                            num_pages = int(info_libro['number_of_pages'])
                            df.at[index, 'num_pages'] = num_pages
                        except:
                            pass
                    if pd.isna(row['publication_year']):
                        try:
                            publication_year = int(info_libro['publish_date'])
                            df.at[index, 'publication_year'] = publication_year
                        except:
                            pass
                else:
                    pass
    print("Done preprocessing num_pages and publication_year")

    # Si algun libro tiene menos de 10 páginas y no es audio, ponerle audio
    for index, row in df.iterrows():
        if row['num_pages'] <= 10 and row['format'] != 'audio':
            df.at[index, 'format'] = 'audio'

    # Si el numero de paginas es nan, ponerle la media de los libros del mismo estilo literario
    ll = df.groupby('estil_literari')['num_pages'].apply(lambda x: x.fillna(x.mean()).astype(int))
    df['num_pages'] = ll.to_list()
    df['num_pages'] = df['num_pages'].astype(int)
    # Si el publication_year es nan, ponerle la media de los libros del mismo estilo literario
    ll = df.groupby('estil_literari')['publication_year'].apply(lambda x: x.fillna(x.mean()).astype(int))
    df['publication_year'] = ll.to_list()
    df['publication_year'] = df['publication_year'].astype(int)

    # LANGUAGE CODE
    # Funció per assignar 'no_identificat' a language_code
    def assigna_noidentificat(valor):
        if valor == 'eng' or valor == 'en-US' or valor == 'en-GB' or valor == 'en-CA' or valor == 'en':
            return 'en'
        if pd.isna(valor) or valor == 'NaN' or valor =='--':
            return 'no_identificat'
        else:
            return valor

    # Aplicar la funció a les columnes isbn i isbn13
    df['language_code'] = df['language_code'].apply(assigna_noidentificat)
    print("Done preprocessing language_code")

    # SERIES
    # Coger los numeros y ponerlos en listas de la variable series
    df['series'] = df['series'].apply(lambda x: np.nan if x == '[]' else x)
    def get_numbers(series):
        if not pd.isna(series):
            series = series.strip("[]").replace("'", "").split(", ")
            series = [int(i.split(" ")[-1]) for i in series]
            return series
        else:
            return np.nan
    df['series'] = df['series'].apply(get_numbers)

    # Get first number of series
    df['series'] = df['series'].apply(lambda x: x[0] if x is not np.nan else np.nan)

    df['series'] = df['series'].astype(str)
    dic_series = df['series'].value_counts().to_dict()
    # put tu nan all series that have 1 book
    df['series'] = df['series'].apply(lambda x: np.nan if dic_series[x] <= 4 else x)
    print("Done preprocessing series")

# Create vector for each book
try:
    llibres = pd.read_pickle(carpeta+pkl_name_ll)
    llibres["vector"]
    get = False
    print("llibres.pkl vector already created. Loading...")
except:
    print("Starting to create llibres.pkl vector")
    get = True
    df = pd.read_pickle(carpeta+pkl_name_ll)

if get:
    # Make dummies for categorical variables and drop the original columns
    llibres_dummies = pd.get_dummies(df, columns=['language_code', 'format', 'series', 'estil_literari', 'complexitat', 'caracteristiques', 'desenvolupament_del_personatge', 'accio_o_reflexio', 'epoca', 'detall_cientific'])
    # Eliminar totes les columnes que no siguin booleanes
    for column in llibres_dummies.columns:
        if llibres_dummies[column].dtype != bool:
            llibres_dummies = llibres_dummies.drop(column, axis=1)

    def scale(vector, min_ant = 0, max_ant = 5, min_nou = 0, max_nou = 1):
        """
        Passar de una valoracio [0-5] a una puntuació [-1-1]
        """
        if isinstance(vector, int):
            vector = np.array([vector])
        if vector.shape[0] > 1:
            min_ant = min(vector)
            #max_ant = max(vector)
        escalador = MinMaxScaler(feature_range=(min_nou, max_nou))
        escalador.fit([[min_ant], [max_ant]])
        return escalador.transform(vector.reshape(-1, 1)).flatten()
    
    scaled_av_rating = scale(df['average_rating'].to_numpy())
    scaled_num_pages = scale(df['num_pages'].to_numpy(), min_ant=0, max_ant=900)
    # All numbers > 1 in scaled_num_pages are set to 1
    scaled_num_pages = np.where(scaled_num_pages > 1, 1, scaled_num_pages)
    scaled_av_rating = np.round(scaled_av_rating, 2)
    scaled_num_pages = np.round(scaled_num_pages, 2)

    llibres_dummies['average_rating'] = scaled_av_rating
    llibres_dummies['num_pages'] = scaled_num_pages

    # Make vector of each book and add it to the dataframe
    vectors = np.array(llibres_dummies).astype(float)
    # Make vectors a list of arrays
    vectors = [np.array(vector) for vector in vectors]
    # Add vectors to dataframe
    df['vector'] = vectors
    llibres = df

    llibres.to_pickle(carpeta+pkl_name_ll)
    llibres.to_csv(carpeta+csv_name_ll, index=False)
    
    print("Done creating llibres.pkl vector")

try:
    casos = pd.read_pickle(carpeta+pkl_name)
    casos["vector"]
    get = False
    print("casos.pkl vector already created. Loading...")
except:
    print("Starting to create casos.pkl vector")
    get = True

if get:
    # Ger user vectors
    casos = pd.read_pickle(carpeta+pkl_name)
    llibres = pd.read_pickle(carpeta+pkl_name_ll)

    def get_attributes(llibres_usuari, val_llibres):
        """
        Aconseguir el vector d'atributs d'usuari a partir dels llibres que ha llegit
        """
        len_vector = len(llibres["vector"].iloc[0])
        vector_usuari = np.zeros(len_vector)
        for ll, val in zip(llibres_usuari, val_llibres):
            vector_usuari += np.array(llibres[llibres["book_id"] == int(ll)]["vector"].iloc[0]) * scale(val)
        np.round(vector_usuari, 1)

        vector_usuari = scale(vector_usuari, min(vector_usuari), max(vector_usuari), -1, 1)

        # Si hay vectores con valores entre -0.01 y 0.01, los ponemos a 0
        for i in range(len(vector_usuari)):
            if vector_usuari[i] < 0.01 and vector_usuari[i] > -0.01:
                vector_usuari[i] = 0

        return np.round(vector_usuari, 4)
    
    vectors = []
    for index, row in casos.iterrows():
        vector_usuari = get_attributes(row['llibres_usuari'], row['val_llibres'])
        vectors.append(vector_usuari)
    casos["vector"] = vectors
    print("Done creating casos.pkl vector")

casos.to_csv(carpeta+csv_name, index=False)
casos.to_pickle(carpeta+pkl_name)
llibres.to_csv(carpeta+csv_name_ll, index=False)
llibres.to_pickle(carpeta+pkl_name_ll)