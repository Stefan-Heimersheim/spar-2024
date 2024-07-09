import igraph as ig
import networkx as nx

def igraph_to_networkx(ig_graph):
    # Create an empty NetworkX graph (or DiGraph if ig_graph is directed)
    if ig_graph.is_directed():
        nx_graph = nx.DiGraph()
    else:
        nx_graph = nx.Graph()

    # Add nodes with attributes
    for v in ig_graph.vs:
        nx_graph.add_node(v.index, **v.attributes())

    # Add edges with attributes
    for e in ig_graph.es:
        source, target = e.tuple
        nx_graph.add_edge(source, target, **e.attributes())

    return nx_graph

def networkx_to_igraph(nx_graph):
    # Determine if the graph is directed
    directed = nx_graph.is_directed()
    
    # Create an empty igraph graph (or DiGraph if nx_graph is directed)
    ig_graph = ig.Graph(directed=directed)

    # Add nodes
    ig_graph.add_vertices(nx_graph.nodes())

    # Add edges
    ig_graph.add_edges(nx_graph.edges())

    # Add node attributes
    for node, data in nx_graph.nodes(data=True):
        for key, value in data.items():
            ig_graph.vs[node][key] = value

    # Add edge attributes
    for source, target, data in nx_graph.edges(data=True):
        eid = ig_graph.get_eid(source, target)
        for key, value in data.items():
            ig_graph.es[eid][key] = value
    
    return ig_graph

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