import csv
import pandas as pd
import random
import numpy as np

fitxer_csv = "../data/books_no_preprocessed.csv"
fitxer_csv_net = "../data/books_clean.csv"

def preprocess_books():
    # Creem una llista buida per emmagatzemar les dades netes
    dades_netes = []

    try:
        with open(fitxer_csv, mode='r', encoding='utf-8') as file:
            reader = csv.reader(file)

            for i, row in enumerate(reader):
                try:
                    # Convertim la fila en un DataFrame i l'afegim a la llista
                    df_line = pd.DataFrame([row])
                    dades_netes.append(df_line)
                except Exception as e:
                    print(f"Error a la línia {i}: {e}")

        # Concatenem totes les files vàlides en un sol DataFrame
        df_net = pd.concat(dades_netes, ignore_index=True)

        # Guardem el DataFrame net en un nou fitxer CSV
        df_net.to_csv(fitxer_csv_net, index=False)
        print(f"Fitxer creat amb èxit: '{fitxer_csv_net}'")
    except Exception as e:
        print(f"S'ha produït un error general: {e}")
        
    # canviar els noms de les columnes per el nom de la primera fila
    df = pd.read_csv(fitxer_csv_net)
    df.columns = df.iloc[0]
    df = df.drop(df.index[0])
    df = df.drop(df.columns[12], axis=1)
    df.to_csv(fitxer_csv_net, index=False)

    # Afegir columnes amb valors aleatoris per cada punt mencionat
    for index, row in df.iterrows():
        df.at[index, 'estil_literari'] = random.choice(["realisme", "romanticisme", "naturalisme", "simbolisme", "modernisme", "realisme magico", "postmodernisme"])
        df.at[index, 'temes_especifics'] = random.choice(["amor", "aventura", "terror", "fantasia", "ciencia ficcio", "historica", "filosofica", "psicologica", "social", "politica", "religiosa", "erotica", "humoristica", "costumista", "negra", "realista", "fantastica", "mitologica", "poetica", "satirica", "biografica", "epica", "didactica", "teatral", "lirica", "epistolar", "dramatica", "epica", "didactica", "teatral", "lirica", "epistolar", "dramatica"])
        df.at[index, 'complexitat'] = random.choice(["baixa", "mitjana", "alta"])
        df.at[index, 'caracteristiques'] = random.choice(["simples", "complexes"])
        df.at[index, 'desenvolupament_del_personatge'] = random.choice(["baix", "mitja", "alt"])
        df.at[index, 'accio_o_reflexio'] = random.choice(["accio", "reflexio"])
        df.at[index, 'longitud'] = random.choice(["curta", "mitjana", "llarga"])
        df.at[index, 'epoca'] = random.choice(["actual", "passada", "futura"])
        df.at[index, 'detall_cientific'] = random.choice(["baix", "mitja", "alta"])

    llibres_dummies = pd.get_dummies(df, columns=['estil_literari', 'temes_especifics', 'complexitat', 'caracteristiques', 'desenvolupament_del_personatge', 'accio_o_reflexio', 'longitud', 'epoca', 'detall_cientific'])
    # Eliminar totes les columnes que no siguin booleanes
    for column in llibres_dummies.columns:
        if llibres_dummies[column].dtype != bool:
            llibres_dummies = llibres_dummies.drop(column, axis=1)
    # Make vector of each book and add it to the dataframe
    vectors = np.array(llibres_dummies).astype(int)
    df["vector"] = vectors.tolist()

    # Guardar la base de dades amb les noves columnes en un nou CSV
    df.to_csv(fitxer_csv_net, index=False)
    df.to_pickle("../data/books_clean.pkl")