function updateInfoPanel(nodeId) {
    const node = graph.getNodeAttributes(nodeId);
    const layer = node.layer;
    const featureNumber = nodeId.split('_')[1];
    
    let html = `<h3>Layer ${layer} Feature ${featureNumber}</h3>`;
    html += `<p>Description placeholder</p>`;
    
    // Parent features
    html += `<h4>Parent features:</h4>`;
    graph.forEachInEdge(nodeId, (edge, attributes, source, target, sourceAttributes, targetAttributes) => {
      const parentLayer = graph.getNodeAttributes(source).layer;
      const parentFeatureNumber = source.split('_')[1];
      html += `<p>Layer ${parentLayer} Feature ${parentFeatureNumber}: Description placeholder. `;
      html += `Similarity: ${attributes.label}</p>`;
    });
    
    // Child features
    html += `<h4>Child features:</h4>`;
    graph.forEachOutEdge(nodeId, (edge, attributes, source, target, sourceAttributes, targetAttributes) => {
      const childLayer = graph.getNodeAttributes(target).layer;
      const childFeatureNumber = target.split('_')[1];
      html += `<p>Layer ${childLayer} Feature ${childFeatureNumber}: Description placeholder. `;
      html += `Similarity: ${attributes.label}</p>`;
    });
    
    document.getElementById('feature-info').innerHTML = html;
  }
  
  // Add event listener for layout change
  document.querySelectorAll('input[name="layer-sort"]').forEach(radio => {
    radio.addEventListener('change', updateLayout);
  });