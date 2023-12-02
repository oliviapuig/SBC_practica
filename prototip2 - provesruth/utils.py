#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#
#          PRACTICA 2 SBC - Sistema de recomenació de llibres
#
# Alumnes: Gomila, Aina
#          Parajó, Ruth
#          Puig, Olívia
#          Ucelayeta, Marc
#
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

### UTILS ###

class Usuari:
    def __init__(self, ide, attributes):
        self.ind = ide

        self.attributes = attributes

        self.llibres_usuari = attributes["llibres_usuari"]
        self.val_llibres = attributes["val_llibres"]

        """self.edat = attributes["edat"]
        self.estil_literari = attributes["estil_literari"]
        self.temas_especifics = attributes["temas_especifics"]
        self.complexitat = attributes["complexitat"]
        self.caracteristiques = attributes["caracteristiques"]
        self.desenvolupament_del_personatge = attributes["desenvolupament_del_personatge"]
        self.accio_o_reflexio = attributes["accio_o_reflexio"]
        self.longitud = attributes["longitud"]
        self.epoca = attributes["epoca"]
        self.detall_cientific = attributes["detall_cientific"]"""

        self.llibres_recomanats = attributes["llibres_recomanats"]
        self.puntuacio_llibres = attributes["puntuacio_llibres"]

        self.vector = self.get_attributes()
        self.cluster= None

    def __str__(self):
        print(f"User {self.get_id()}")
        for key, value in self.attributes.items():
            print(f"{key}: {value}")
        return ""
    
    def get_id(self):
        return self.ind
    
    def scale(self, vector, min_ant = 0, max_ant = 5, min_nou = -1, max_nou = 1):
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
    
    def get_attributes(self):
        """
        Aconseguir el vector d'atributs d'usuari a partir dels llibres que ha llegit
        """
        llibres = pd.read_pickle("../data/books_clean.pkl")
        len_vector = len(llibres["vector"].iloc[0])
        vector_usuari = np.zeros(len_vector)
        for ll, val in zip(self.llibres_usuari, self.val_llibres):
            vector_usuari += np.array(llibres[llibres["isbn13"] == ll]["vector"].iloc[0]) * self.scale(val)
        #vector_usuari = self.scale(vector_usuari)
        #print("Vector usuari escalat: ", vector_usuari)

        # Si hay vectores con valores entre -0.01 y 0.01, los ponemos a 0
        for i in range(len(vector_usuari)):
            if vector_usuari[i] < 0.01 and vector_usuari[i] > -0.01:
                vector_usuari[i] = 0

        return np.round(vector_usuari, 1)
    ### afegir ruth's metric