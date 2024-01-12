# # SBC - Pràctica CBR
# 
# - Aina Gomila
# - Ruth Parajó
# - Olivia Puig
# - Marc Ucelayeta

from cbr import CBR
import pickle
import pandas as pd
import numpy as np
import warnings
import random

warnings.filterwarnings("ignore")


cases =pd.read_pickle('./data/casos.pkl')
books = pd.read_pickle('./data/llibres.pkl')

with open('./data/clustering/model_clustering_casos.pkl', 'rb') as arxiu:
    clustering = pickle.load(arxiu)

cbr = CBR(cases,clustering,books)

nou_cas = cbr.inicia("prototip3/joc_de_proves.txt")
recomanacio = cbr.recomana(nou_cas)
