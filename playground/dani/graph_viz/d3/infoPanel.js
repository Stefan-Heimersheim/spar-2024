function updateInfoPanel(d) {
    const infoPanel = document.getElementById("node-info");
    if (d) {
        let html = `<h3>Layer ${d.layer} Feature ${d.feature}</h3>`;
        html += `<p>${d.explanation}</p>`;
        html += "<h4>Connected Features:</h4>";

        graph.links.forEach(link => {
            if (link.source === d || link.target === d) {
                const connectedNode = link.source === d ? link.target : link.source;
                html += `<p>Layer ${connectedNode.layer} Feature ${connectedNode.feature}: ${connectedNode.explanation}</p>`;
                html += `<p>Similarity: ${link.similarity.toFixed(2)}</p>`;
            }
        });

        infoPanel.innerHTML = html;
    } else {
        infoPanel.innerHTML = "Click on a node to see information";
    }
}