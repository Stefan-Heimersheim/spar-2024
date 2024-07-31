import plotly.graph_objects as go
import networkx as nx

def show_explanation_graph(graph: nx.DiGraph) -> None:
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
    fig.show()
