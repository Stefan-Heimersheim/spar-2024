import plotly.graph_objects as go
import networkx as nx
import torch
import numpy as np
import einops
import pickle
import json

def show_explanation_graph(graph: nx.DiGraph, show: bool) -> None:
    layout = nx.multipartite_layout(graph, subset_key='layer')
    
    # Each edge is an individual trace, otherwise the width must be the same
    edge_traces = [go.Scatter(
        x=[layout[v][0], layout[w][0]],
        y=[layout[v][1], layout[w][1]],
        line=dict(width=5 * attr['similarity'], color='red'),
        mode='lines',
        # TODO: Add intermediate points on edge for hover info
    ) for v, w, attr in graph.edges(data=True)]

    node_x, node_y = list(zip(*[layout[node] for node in graph.nodes()]))
    feat = [node for node in graph.nodes]
    node_colors = ['blue' if attr.get('is_downstream', False) else 'green' for _, attr in graph.nodes(data=True)]
    hover = [f'Explanation: {attr["explanation"]}' for _, attr in graph.nodes(data=True)]

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        hoverinfo='text',
        marker=dict(size=10, color=node_colors),
        text=feat,
        textfont=dict(size=8),
        hovertext=hover,
        textposition='bottom center'
    )

    fig = go.Figure(data=[*edge_traces, node_trace],
                layout=go.Layout(
                    title='SAE Feature Interaction Graph',
                    titlefont_size=16,
                    showlegend=False,
                    margin=dict(b=0,l=0,r=0,t=30),
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                    )
    if show:
        fig.show()

    return fig


def get_active_features(model, saes, tokens, activation_threshold, hook_name='hook_resid_pre', artefacts_folder='../../artefacts'):
    n_layers = model.cfg.n_layers
    n_features = saes[0].cfg.d_sae
    batch_size, context_size = tokens.shape

    max_activations = torch.tensor(np.load(f'{artefacts_folder}/max_sae_activations/res_jb_max_sae_activations_17.5M.npz')['arr_0'])
    activation_threshold = (activation_threshold * max_activations).unsqueeze(2).unsqueeze(3)

    sae_activations = torch.empty(n_layers, batch_size, context_size, n_features)

    def retrieval_hook(activations, hook):
        layer = hook.layer()

        sae_activations[layer] = saes[layer].encode(activations)

    model.add_hook(lambda name: name.endswith(f".{hook_name}"), retrieval_hook)

    with torch.no_grad():
        model.run_with_hooks(tokens)

        # Now we can use sae_activations
        return einops.rearrange((einops.rearrange(sae_activations, 'n_layers n_samples n_tokens n_features -> n_layers n_features n_samples n_tokens') > activation_threshold)[:, :, :, -1].bool(), 'n_layers n_features n_samples -> n_samples n_layers n_features')


def build_graph(features, similarities, sae_name: str = 'res_jb_sae', verbose=False):
    # Convert matrix into graph
    if verbose:
        print('Extracting nodes from feature matrix...')
    n_samples, n_layers, n_features = features.shape
    assert n_samples == 1, 'Please provide a single sample only!'

    nodes = [features[0, layer].nonzero().flatten().tolist() for layer in range(n_layers)]

    if verbose:
        print('Adding nodes to graph...')
    graph = nx.DiGraph()
    for layer, features in enumerate(nodes):
        graph.add_nodes_from([(f'{layer}_{feature}', {'layer': layer, 'feature': feature}) for feature in features])

    if verbose:
        print('Adding edges to graph...')
    for layer, (features_from, features_to) in enumerate(zip(nodes, nodes[1:])):
        graph.add_edges_from([(f'{layer}_{out_feature}', f'{layer+1}_{in_feature}', {'similarity': abs(similarities[layer, out_feature, in_feature])}) for out_feature in features_from for in_feature in features_to])

    if verbose:
        print('Loading explanations...')
    with open(f'../../artefacts/explanations/{sae_name}_explanations.pkl', 'rb') as f:
        explanations = pickle.load(f)

    if verbose:
        print('Adding explanations to graph...')
    for node, attr in graph.nodes(data=True):
        graph.nodes[node]['explanation'] = explanations[attr['layer']][attr['feature']]

    return graph


def get_active_feature_graph_for_prompt(model, saes, prompt, similarities, activation_threshold_2, artefacts_folder='../../artefacts', verbose=False):
    if verbose:
        print('Running model...')
    tokens = model.to_tokens(prompt)

    if verbose:
        print('Extracting active features on last token...')
    features = get_active_features(model, saes, tokens, activation_threshold_2, artefacts_folder=artefacts_folder)

    if verbose:
        print('Building graph...')
    graph = build_graph(features, similarities, verbose=verbose)

    if verbose:
        print('Done!')

    return graph


def save_graph_to_json(graph, filename):
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.float32):
                return float(obj)
            # Add more conditions here for other numpy types, if needed
            return super(NumpyEncoder, self).default(obj)

    with open(filename, 'w') as f:
        json.dump(nx.node_link_data(graph), f, cls=NumpyEncoder)