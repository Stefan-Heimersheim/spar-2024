function updateInfoPanel(node) {
    const infoPanel = document.getElementById('feature-details');
    if (!node) {
        infoPanel.innerHTML = '<p>Click on a node to see information</p>';
        return;
    }

    const [layer, feature] = node.id.split('_');
    const neighbors = nodeNeighbors.get(node.id);

    let html = `
        <h3>Layer ${layer}, Feature ${feature}</h3>
        <p>${node.explanation}</p>
        <h4>Connected Features:</h4>
    `;

    if (neighbors.nodes.size === 0) {
        html += '<p>None</p>';
    } else {
        // Create an array of neighbor objects with their details
        const neighborDetails = Array.from(neighbors.nodes).map(neighborId => {
            const neighborNode = graph.nodes.find(n => n.id === neighborId);
            const [neighborLayer, neighborFeature] = neighborId.split('_');
            const similarity = graph.links.find(l => 
                (l.source === node.id && l.target === neighborId) || 
                (l.source === neighborId && l.target === node.id)
            ).similarity;
            const link = `https://www.neuronpedia.org/gpt2-small/${neighborLayer}-res-jb/${neighborFeature}`;
            return { neighborLayer, neighborFeature, neighborNode, similarity, link };
        });

        // Sort neighbors by similarity in descending order
        neighborDetails.sort((a, b) => b.similarity - a.similarity);

        html += `
        <table>
            <tr><th>Layer</th><th>Feature</th><th>Explanation</th><th>Similarity</th><th>Link</th></tr>
        `;

        neighborDetails.forEach(({ neighborLayer, neighborFeature, neighborNode, similarity, link }) => {
            html += `
                <tr data-node-id="${neighborNode.id}" class="info-table-row">
                    <td>${neighborLayer}</td>
                    <td>${neighborFeature}</td>
                    <td>${neighborNode.explanation}</td>
                    <td class="similarity">${similarity.toFixed(2)}</td>
                    <td><a href="${link}" target="_blank" rel="noopener noreferrer">
                        <img src="external-link-icon.png" alt="Open in new tab" style="width: 16px; height: 16px;">
                    </a></td>
                </tr>
            `;
        });

        html += '</table>';
    }

    infoPanel.innerHTML = html;

    // Add event listeners to table rows
    const tableRows = infoPanel.querySelectorAll('table tr[data-node-id]');
    tableRows.forEach(row => {
        row.addEventListener('mouseenter', () => {
            const nodeId = row.getAttribute('data-node-id');
            const hoveredNode = graph.nodes.find(n => n.id === nodeId);
            highlightOneNode(hoveredNode, 'hovered');
        });
        row.addEventListener('mouseleave', () => {
            const nodeId = row.getAttribute('data-node-id');
            const hoveredNode = graph.nodes.find(n => n.id === nodeId);
            handleNodeLeave(null, hoveredNode);
            // resetGraphStyles();
        });
    });
}