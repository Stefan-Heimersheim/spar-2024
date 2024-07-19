import igraph as ig
import networkx as nx

# igraph provides some very nice converters!

def igraph_to_networkx(ig_graph):
    
    return ig_graph.to_networkx()

def networkx_to_igraph(nx_graph):
   G_ig = ig.Graph.from_networkx(nx_graph)
   G_ig.vs['name'] = list(nx_graph.nodes())
   return G_ig

# Example usage
if __name__ == "__main__":
    # Create an example NetworkX graph
    nx_graph = nx.DiGraph()
    nx_graph.add_nodes_from([(0, {"name": "A"}), (1, {"name": "B"}), (2, {"name": "C"})])
    nx_graph.add_edges_from([(0, 1, {"weight": 1.0}), (1, 2, {"weight": 2.0}), (0,2, {"weight":3.0})])

    # Convert to igraph graph
    ig_graph = networkx_to_igraph(nx_graph)

    # Print the igraph graph to verify
    print(ig_graph,"\n---")
    print(ig_graph.vs['name'],"\n---")
    print(ig_graph.es['weight'],"\n---")