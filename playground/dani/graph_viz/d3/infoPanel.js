function updateInfoPanel(d) {
    const infoPanel = document.getElementById("node-info");
    if (d) {
        let html = `<h3>Layer ${d.layer} Feature ${d.feature}</h3>`;
        html += `<p>${d.explanation}</p>`;
        html += "<h4>Connected Features:</h4>";

        console.log(graph.links);
        graph.links.forEach(link => {
            console.log(link);
            if (link.source === d.id || link.target === d.id) {
                const connectedNodeId = link.source === d.id ? link.target : link.source;
                const connectedNode = graph.nodes.find(node => node.id === connectedNodeId);
                html += `<p>Layer ${connectedNode.layer} Feature ${connectedNode.feature}: ${connectedNode.explanation}</p>`;
                html += `<p>Similarity: ${link.similarity.toFixed(2)}</p>`;
            }
        });

        infoPanel.innerHTML = html;
    } else {
        infoPanel.innerHTML = "Click on a node to see information";
    }
}