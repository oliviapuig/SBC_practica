Title
===
`SBC - Pràctica CBR`
## Paper Information 
- Authors:  `Aina Gomila`,`Ruth Parajó`,`Olivia Puig`,`Marc Ucelayeta`

## Install & Dependence
- python
- pandas
- numpy
- pickle
- sklearn

## Use
- Per ús individual
  ```
  python prototip3/prototip3.py
  ```
- Per ús avaluatori
  ```
  python prototip3/prototip3_avaluacio_temporal.py
  python prototip3/prototip3_avaluacio_cbr.py
  ```

## Directory Hierarchy
```
|—— data
|    |—— README.md
|    |—— casos.csv
|    |—— casos.pkl
|    |—— casos_actualitzat.pkl
|    |—— casos_dataset.py
|    |—— clustering
|        |—— clustering.ipynb
|        |—— dendograma.png
|        |—— model_clustering_casos.pkl
|        |—— model_clustering_casos_actualitzat.pkl
|        |—— model_clustering_llibres.pkl
|    |—— dades_actualitzades.ipynb
|    |—— llibres.csv
|    |—— llibres.pkl
|    |—— scrapping.ipynb
|—— plots
|    |—— barplot_bestseller.png
|    |—— barplot_genres.png
|    |—— books_rated_after.png
|    |—— books_rated_before.png
|    |—— colze.png
|    |—— hist_average_rating.png
|    |—— hist_ratings_count.png
|    |—— hist_ratings_count_limited_20000.png
|    |—— kmeans_pca_casos.png
|    |—— kmeans_pca_casos_actualitzada.png
|    |—— kmeans_pca_casos_actualitzat.png
|    |—— kmeans_pca_llibres.png
|    |—— numero_posicions_vector_abans.png
|    |—— numero_posicions_vector_despres.png
|    |—— pca_casos_cluster_3d.html
|    |—— pca_llibres.png
|    |—— pca_llibres_cluster_3d.html
|    |—— pca_llibres_cluster_after.png
|    |—— pca_llibres_cluster_before.jpg
|    |—— plots.ipynb
|    |—— puntuacions_recomanacions.png
|    |—— rating_distribution.png
|    |—— temps_recomanacio.png
|    |—— tsne_llibres.png
|—— prototip0
|    |—— cbr.py
|    |—— prototip0.ipynb
|    |—— utils.py
|—— prototip1
|    |—— canvis.txt
|    |—— cbr.py
|    |—— prototip1.ipynb
|    |—— utils.py
|—— prototip2
|    |—— cbr.py
|    |—— prototip2.py
|    |—— utils.py
|—— prototip3
|    |—— cbr.py
|    |—— prototip3.py
|    |—— prototip3_avaluacio_cbr.py
|    |—— prototip3_avaluacio_temporal.py
```
## Content
### data
Conté les bases de dades utilitzades com a casos i llibres i el py per la generació sintètica d'aquestes.
- Clustering
  Inclou els models de clustering 
  

### plots
Inclouen tots plots realitzats per l'analisi del recomanador.

### prototip0
Prototip inicial del recomanador, conté el sistema cbr i l'execució del prototip.
### prototip1
Prototip 1 inicial del recomanador, conté el sistema cbr i l'execució del prototip.

### prototip2
Prototip 2 del recomanador, conté el sistema cbr i l'execució del prototip.
### prototip3
Prototip final del recomanador, conté el sistema cbr i l'execució del prototip.

## License

Universitat Politècnica de Catalunya