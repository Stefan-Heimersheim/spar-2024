# %%
!pip install graphistry
# %%
!pip install pandas
# %%
import json
import pandas as pd
import graphistry

# Load the JSON file
with open('playground/dani/graph_viz/sample_graph.json', 'r') as file:
    data = json.load(file)
# %%

# Create DataFrames for nodes and edges
nodes_df = pd.DataFrame(data['nodes'])
edges_df = pd.DataFrame(data['links'])

# Rename columns to match Graphistry's expected format
edges_df = edges_df.rename(columns={'source': 'src', 'target': 'dst'})
# %%
nodes_df.head()
# %%
edges_df.head()
# %%
# Initialize Graphistry (you might need to set your API key here)
# graphistry.register(api=3, protocol="https", server="hub.graphistry.com", username="username", password="password")

# Create the plot
plot = graphistry.bind(source="src", destination="dst", node="id")\
    .nodes(nodes_df)\
    .edges(edges_df)\
    .settings(url_params={'play': 7000})\
    .plot()
# %%
plot.settings()
# %%
plot.settings(url_params={'play': 7000})
# %%
# Display the plot
print(plot)
# %%
