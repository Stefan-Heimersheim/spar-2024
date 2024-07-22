import networkx as nx
import igraph as ig
import matplotlib.pyplot as plt

def nx_graph_from_tensor(tensor):
    assert len(tensor.shape) == 3
    assert tensor.shape[0] == tensor.shape[1]

    # Note that if n_layers == 11, then there ACTUALLY 12 layers. Its represented in this form
    # because it is describing layer PAIRS
    d_sae, _, n_layers = tensor.shape

    # Here we change it to the correct value
    n_layers += 1

    G = nx.DiGraph()
    
    # Adding nodes
    for feature in range(d_sae):
        for layer in range(n_layers):
            G.add_node(f"layer_{layer}_feature_{feature}")
    
    # Adding edges
    for start_layer in range(n_layers - 1):
        for start_feature in range(d_sae):
            for end_feature in range(d_sae):
                if tensor[start_feature, end_feature, start_layer] != 0:
                    G.add_edge(f"layer_{start_layer}_feature_{start_feature}", f"layer_{start_layer+1}_feature_{end_feature}", weight=tensor[start_feature, end_feature, start_layer])

    return G


def igraph_from_tensor(tensor):
    assert len(tensor.shape) == 3
    assert tensor.shape[0] == tensor.shape[1]

    # Note that if n_layers == 11, then there ACTUALLY 12 layers. Its represented in this form
    # because it is describing layer PAIRS
    d_sae, _, n_layers = tensor.shape

    # Here we change it to the correct value
    n_layers += 1  

    # Create the igraph graph
    g = ig.Graph(directed=True)

    # Add nodes to the graph
    for layer in range(n_layers):
        for feature in range(d_sae):
            g.add_vertex(name=f"layer_{layer}_feature_{feature}")


    # Add directed edges with weights between consecutive layers
    for layer in range(n_layers - 1):
        for source_node in range(d_sae):
            for target_node in range(d_sae):
                source = g.vs.find(name=f'layer_{layer}_node_{source_node}')
                target = g.vs.find(name=f'layer_{layer + 1}_node_{target_node}')
                weight = tensor[source_node, target_node, layer]
                g.add_edge(source, target, weight=weight)

        # Adding edges
    for start_layer in range(n_layers - 1):
        for start_feature in range(d_sae):
            for end_feature in range(d_sae):
                if tensor[start_feature, end_feature, start_layer] != 0:
                    source = g.vs.find(name=f'layer_{layer}_node_{source_node}')
                    target = g.vs.find(name=f'layer_{layer + 1}_node_{target_node}')
                    weight = tensor[source_node, target_node, layer]
                    g.add_edge(source, target, weight=weight)



def plot_nx_graph_with_mpl(G, num_layers=12, num_features=5, filename="plot.png", threshold=None):
    # displays networkx graph, G, using matplotlib
    # Compute positions for all nodes
    # Generate positions for the nodes
    pos = {}
    layer_height = 1
    layer_spacing = 10

    for layer in range(num_layers):
        for feature in range(num_features):
            pos[f"layer_{layer}_feature_{feature}"] = (layer * layer_spacing, feature * layer_height)

    # Plot the graph
    plt.figure(figsize=(int(num_layers * 2.5), int(num_features * .25) + 1))

    if threshold != None:
        filtered_edges = [(u, v) for u, v, d in G.edges(data=True) if d['weight'] > threshold]
        nx.draw_networkx_nodes(G, pos, node_size=100, node_color='blue')
        #nx.draw_networkx_labels(G, pos, font_size=15, font_family='sans-serif', font_color='black')
        nx.draw_networkx_edges(G, pos, edgelist=filtered_edges, edge_color='black', arrows=False)
    else:
        nx.draw(G, pos, node_size=100, edge_color='lightgrey', alpha=0.5, node_color="blue", arrows=False)

    plt.title("Layered Graph Visualization")
    plt.savefig(f"plots/{filename}", dpi=300, bbox_inches='tight')
    plt.close()