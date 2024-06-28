# %%

# %%
# %%
# Create a simple graph
import igraph as ig
import matplotlib.pyplot as plt

%matplotlib inline

# Create a simple graph
g = ig.Graph(edges=[(0, 1), (1, 2), (2, 3), (3, 0)], directed=False)

# Add vertex labels
g.vs["label"] = ["A", "B", "C", "D"]

# Plot the graph
layout = g.layout("circle")
ig.plot(
    g,
    layout=layout,
    bbox=(300, 300),
    vertex_size=30,
    vertex_label=g.vs["label"],
    edge_width=2
)

# To display the plot using matplotlib
plt.show()
# %%
