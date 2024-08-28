function updateInfoPanel(node) {
    const infoPanel = document.getElementById('feature-details');
    if (!node) {
        infoPanel.innerHTML = '<p>Click on a node to see information</p>';
        return;
    }

    const [layer, feature] = node.id.split('_');
    const neighbors = nodeNeighbors.get(node.id);

    let html = `
        <h3>Layer ${layer} Feature ${feature}</h3>
        <p>${node.explanation}</p>
        <h4>Connected Features:</h4>
    `;

    if (neighbors.nodes.size === 0) {
        html += '<p>None</p>';
    } else {
        html += `
        <table>
            <tr><th>Layer</th><th>Feature</th><th>Explanation</th><th>Similarity</th></tr>
        `;

        neighbors.nodes.forEach(neighborId => {
            const neighborNode = graph.nodes.find(n => n.id === neighborId);
            const [neighborLayer, neighborFeature] = neighborId.split('_');
            const similarity = graph.links.find(l => 
                (l.source === node.id && l.target === neighborId) || 
                (l.source === neighborId && l.target === node.id)
            ).similarity;

            html += `
                <tr>
                    <td>${neighborLayer}</td>
                    <td>${neighborFeature}</td>
                    <td>${neighborNode.explanation}</td>
                    <td class="similarity">${similarity.toFixed(2)}</td>
                </tr>
            `;
        });

        html += '</table>';
    }

    infoPanel.innerHTML = html;
}