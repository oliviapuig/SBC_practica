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

class Usuari:
    def __init__(self, ind, attributes):
        self.ind = ind
        self.attributes = attributes
        self.estils_literaris = attributes["estils_literaris"]
        self.temes_de_llibres = attributes.get("temes_de_llibres", None)
        self.complexitat = attributes.get("complexitat", None)
        self.demografia = attributes.get("demografia", None)
        self.situacio = attributes.get("situacio", None)
        self.estat_civil = attributes.get("estat_civil", None)

        # Afegeix altres atributs segons necessitat
        self.vector = []


        self.llibres_recomanats = attributes["llibres_recomanats"]
        self.puntuacio_llibres = attributes["puntuacio_llibres"]
    
    def __str__(self):
        print(f"User {self.ind}")
        for key, value in self.attributes.items():
            print(f"{key}: {value}")
        return ""
    