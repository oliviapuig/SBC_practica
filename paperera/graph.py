import pandas as pd
import networkx as nx
import json

data = pd.read_pickle('data/llibres.pkl')

llibres = []
for i in range(len(data)):
    # Afegeix una tupla amb el llibre i els seus similars
    llibres.append((data['book_id'][i], data['similar_books'][i]))

G = nx.DiGraph()

for book_id, similar_books in llibres:
    for similar_book_id in similar_books:
        G.add_edge(book_id, similar_book_id)

import matplotlib.pyplot as plt

# Print the number of nodes and edges in original graph
print("Original Graph")
print("Nodes:", len(G.nodes()))
print("Edges:", len(G.edges()))

nx.draw(G, with_labels=False)
plt.show()