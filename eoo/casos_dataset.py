json_name = 'data.json'
pkl_name = 'casos.pkl'
csv_name = 'casos.csv'
carpeta = 'eoo'

import requests
import gzip
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# URL del archivo JSON comprimido
url = 'https://datarepo.eng.ucsd.edu/mcauley_group/gdrive/goodreads/goodreads_reviews_dedup.json.gz'

# Realizar la solicitud GET al servidor
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
plt.savefig(f'{carpeta}/rating_distribution.png')

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
# x: each user
# y: number of books rated
plt.figure(figsize=(10, 6))
plt.xlabel('user_id')
plt.ylabel('Number of books rated')
plt.title('Number of books rated by each user')
plt.plot(df_aux['user_id'], df_aux['books'].apply(lambda x: len(x)))
plt.savefig(f'{carpeta}/books_rated_before.png')

min_books = 10
max_books = 20

# Remove users that have rated less than 10 books and more than 50
df_aux = df_aux[df_aux['books'].apply(lambda x: len(x) >= min_books and len(x) <= max_books)]
df_aux = df_aux.reset_index(drop=True)

print(f"Dataset filtered with users with more than {min_books} and less than {max_books} books reviewed. Unique users:", len(df_aux))

# Plot how many books each user has rated and save to eoo/books_rated_after.png
# x: each user
# y: number of books rated
plt.figure(figsize=(10, 6))
plt.xlabel('user_id')
plt.ylabel('Number of books rated')
plt.title('Number of books rated by each user')
plt.plot(df_aux['user_id'], df_aux['books'].apply(lambda x: len(x)))
plt.savefig(f'{carpeta}/books_rated_after.png')

# For each user get 3 last books and their ratings and put them in a new column "llibres_recomanata" i "puntuacions_llibres". Then remove the 3 books from the list of books rated by the user.
df_aux['llibres_recomanats'] = df_aux['books'].apply(lambda x: x[-3:])
df_aux['puntuacions_llibres'] = df_aux['ratings'].apply(lambda x: x[-3:])
df_aux['books'] = df_aux['books'].apply(lambda x: x[:-3])
df_aux['ratings'] = df_aux['ratings'].apply(lambda x: x[:-3])

print("Done creating new columns.")

# Change "books" and "ratings" columns to "llibres_usuari" and "val_llibres"
df_aux = df_aux.rename(columns={'books': 'llibres_usuari', 'ratings': 'val_llibres'})

set_llibres = set()
for llibres in df_aux['llibres_usuari']:
    set_llibres.update(llibres)
for llibres in df_aux['llibres_recomanats']:
    set_llibres.update(llibres)

# URL del archivo JSON comprimido
url = 'https://datarepo.eng.ucsd.edu/mcauley_group/gdrive/goodreads/goodreads_books.json.gz'

# Realizar la solicitud GET al servidor
response = requests.get(url, stream=True)

# Verificar si la solicitud fue exitosa (código de estado 200)
if response.status_code == 200:
    # Get the books that their book_id is in set_llibres
    data = []
    with gzip.GzipFile(fileobj=response.raw) as f:
        for line in f:
            line = json.loads(line)
            if line['book_id'] in set_llibres:
                data.append(line)

    print("JSON creat.")
else:
    print(f"Error al descargar el archivo. Código de estado: {response.status_code}")

llibres = pd.DataFrame(primeras_500_filas)


# For each user and each book inside "llibres_usuari" and "llibres_recomanats" get the book_id and 


def scale(vector, min_ant = 0, max_ant = 5, min_nou = -1, max_nou = 1):
    """
    Passar de una valoracio [0-5] a una puntuació [-1-1]
    """
    if isinstance(vector, int):
        vector = np.array([vector])
    if vector.shape[0] > 1:
        min_ant = min(vector)
        max_ant = max(vector)
    escalador = MinMaxScaler(feature_range=(min_nou, max_nou))
    escalador.fit([[min_ant], [max_ant]])
    return escalador.transform(vector.reshape(-1, 1)).flatten()

llibres = pd.read_pickle("data/books_clean.pkl")

def get_attributes(llibres_usuari, val_llibres):
    """
    Aconseguir el vector d'atributs d'usuari a partir dels llibres que ha llegit
    """
    len_vector = len(llibres["vector"].iloc[0])
    vector_usuari = np.zeros(len_vector)
    for ll, val in zip(llibres_usuari, val_llibres):
        vector_usuari += np.array(llibres[llibres["isbn13"] == ll]["vector"].iloc[0]) * scale(val)
    vector_usuari = scale(vector_usuari)
    #print("Vector usuari escalat: ", vector_usuari)

    # Si hay vectores con valores entre -0.01 y 0.01, los ponemos a 0
    for i in range(len(vector_usuari)):
        if vector_usuari[i] < 0.01 and vector_usuari[i] > -0.01:
            vector_usuari[i] = 0

    return np.round(vector_usuari, 1)

# For each user get the vector of attributes and put it in a new column "vector"
df_aux['vector'] = df_aux.apply(lambda x: get_attributes(x['llibres_usuari'], x['val_llibres']), axis=1)

print("Done creating vector column.")

df_aux.to_pickle(pkl_name)
df_aux.to_csv(csv_name, index=False)

print("Saved to pickle and csv.")