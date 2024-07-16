import networkx as nx

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
                    G.add_edge(f"layer_{start_layer}_feature_{start_feature}", f"layer_{start_layer+1}_feature_{end_feature}")


    
    
    return G