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

### UTILS ###

from sklearn.preprocessing import OneHotEncoder
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import DistanceMetric

class Usuari:
    def __init__(self, ind, attributes):
        self.ind = ind
        self.attributes = attributes
        self.edat = attributes["edat"]
        self.estil_literari = attributes["estil_literari"]
        self.temas_especifics = attributes["temas_especifics"]
        self.complexitat = attributes["complexitat"]
        self.caracteristiques = attributes["caracteristiques"]
        self.desenvolupament_del_personatge = attributes["desenvolupament_del_personatge"]
        self.accio_o_reflexio = attributes["accio_o_reflexio"]
        self.longitud = attributes["longitud"]
        self.epoca = attributes["epoca"]
        self.detall_cientific = attributes["detall_cientific"]

        self.vector = []

        self.llibres_recomanats = []
        self.puntuacio_llibres = []
    
    def __str__(self):
        print(f"User {self.ind}")
        for key, value in self.attributes.items():
            print(f"{key}: {value}")
        return ""