import pandas as pd
import networkx as nx
import json

with open('eoo.json', 'r') as file:
    data = json.load(file)

books = [(book['book_id'], book['similar_books']) for book in data]

G = nx.DiGraph()

for book_id, similar_books in books:
    for similar_book_id in similar_books:
        G.add_edge(book_id, similar_book_id)

import matplotlib.pyplot as plt

nx.draw(G, with_labels=True)
plt.show()