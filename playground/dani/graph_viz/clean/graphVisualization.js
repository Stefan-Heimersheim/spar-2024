function initializeGraph(data) {
    data.nodes.forEach((node, index) => {
      graph.addNode(node.id, {
        x: Math.random(),
        y: Math.random(),
        size: 5,
        label: node.id,
        layer: node.layer,
        originalIndex: index,
        color: defaultNodeColor
      });
    });
  
    data.links.forEach((link) => {
      if (!graph.hasEdge(link.source, link.target)) {
        graph.addEdge(link.source, link.target, {
          size: 4 * link.similarity,
          label: `${link.measure}: ${link.similarity.toFixed(2)}`,
          color: defaultEdgeColor
        });
      }
    });
  }
  
  function initializeSigma() {
    const container = document.getElementById('sigma-container');
    
    sigmaInstance = new Sigma(graph, container, {
      renderEdgeLabels: false,
      allowWheelZoom: false,
      renderLabels: false,
      labelRenderedSizeThreshold: -Infinity,
      labelDensity: 0.07,
      labelGridCellSize: 60,
      labelFont: "Arial",
      labelWeight: "bold",
      labelColor: {
        color: "#000000",
        attribute: null,
        interpolation: null,
      },
      labelSize: 14,
      labelPlacement: "center",
      renderNodes: renderCustomNodes
    });
  
    sigmaInstance.on('enterNode', handleNodeHover);
    sigmaInstance.on('leaveNode', handleNodeLeave);
  }
  
  function renderCustomNodes(context, nodes, camera) {
    nodes.forEach((node) => {
      const { x, y } = camera.framedGraphToViewport(node);
      context.fillStyle = node.color;
      context.beginPath();
      context.arc(x, y, 5, 0, Math.PI * 2);
      context.fill();
  
      if (node.highlighted || node.isNeighbor) {
        const tooltip = createNodeTooltip(node.key);
        context.fillStyle = "rgba(0, 0, 0, 0.8)";
        context.fillRect(x + 10, y - 20, 70, 25);
        context.fillStyle = "#ffffff";
        context.font = "12px Arial";
        context.fillText(tooltip, x + 15, y - 5);
      }
    });
  }
  
  function createNodeTooltip(node) {
    const featureIndex = node.split('_')[1];
    return `Index: ${featureIndex}`;
  }
  
  function getConnectedElements(nodeId) {
    const connectedNodes = new Set([nodeId]);
    const connectedEdges = new Set();
  
    graph.forEachEdge((edge, attributes, source, target) => {
      if (source === nodeId || target === nodeId) {
        connectedNodes.add(source);
        connectedNodes.add(target);
        connectedEdges.add(edge);
      }
    });
  
    return { nodes: connectedNodes, edges: connectedEdges };
  }
  
  function handleNodeHover(event) {
    const nodeId = event.node;
    const { nodes, edges } = getConnectedElements(nodeId);
  
    graph.forEachNode((node, attributes) => {
      if (nodes.has(node)) {
        graph.setNodeAttribute(node, 'color', highlightColor);
        graph.setNodeAttribute(node, 'highlighted', node === nodeId);
        graph.setNodeAttribute(node, 'isNeighbor', node !== nodeId);
      } else {
        graph.setNodeAttribute(node, 'color', defaultNodeColor);
        graph.setNodeAttribute(node, 'highlighted', false);
        graph.setNodeAttribute(node, 'isNeighbor', false);
      }
    });
  
    graph.forEachEdge((edge, attributes) => {
      graph.setEdgeAttribute(edge, 'color', edges.has(edge) ? highlightColor : defaultEdgeColor);
    });
  
    updateInfoPanel(event.node);
    sigmaInstance.refresh();
  }
  
  function handleNodeLeave() {
    graph.forEachNode((node) => {
      graph.setNodeAttribute(node, 'color', defaultNodeColor);
      graph.setNodeAttribute(node, 'highlighted', false);
      graph.setNodeAttribute(node, 'isNeighbor', false);
    });
  
    graph.forEachEdge((edge) => {
      graph.setEdgeAttribute(edge, 'color', defaultEdgeColor);
    });
  
    document.getElementById('feature-info').innerHTML = '<h2>Feature Information</h2><p>Hover over a node to see information</p>';
    sigmaInstance.refresh();
  }
  
  function updateLayout() {
    const sortMethod = document.querySelector('input[name="layer-sort"]:checked').value;
    const containerWidth = document.getElementById('sigma-container').offsetWidth;
    const containerHeight = document.getElementById('sigma-container').offsetHeight;
  
    const nodesByLayer = {};
    graph.forEachNode((node, attributes) => {
      if (!nodesByLayer[attributes.layer]) {
        nodesByLayer[attributes.layer] = [];
      }
      nodesByLayer[attributes.layer].push({ id: node, attributes });
    });
  
    if (sortMethod === 'minimize-crossings') {
      Object.keys(nodesByLayer).forEach(layer => {
        nodesByLayer[layer].sort((a, b) => {
          const aConnections = graph.degree(a.id);
          const bConnections = graph.degree(b.id);
          return bConnections - aConnections;
        });
      });
    } else {
      Object.keys(nodesByLayer).forEach(layer => {
        nodesByLayer[layer].sort((a, b) => {
          const aIndex = parseInt(a.id.split('_')[1]);
          const bIndex = parseInt(b.id.split('_')[1]);
          return aIndex - bIndex;
        });
      });
    }
  
    const layerCount = Object.keys(nodesByLayer).length;
    const topMargin = 120;
    const bottomMargin = 40;
    const layerSpacing = (containerHeight - topMargin - bottomMargin) / (layerCount - 1);
    const horizontalPadding = containerWidth * 0.05;
  
    Object.keys(nodesByLayer).forEach((layer, layerIndex) => {
      const nodesInLayer = nodesByLayer[layer].length;
      const availableWidth = containerWidth - 2 * horizontalPadding;
      const nodeSpacing = availableWidth / (nodesInLayer - 1 || 1);
      nodesByLayer[layer].forEach((node, index) => {
        graph.setNodeAttribute(node.id, 'x', horizontalPadding + index * nodeSpacing);
        graph.setNodeAttribute(node.id, 'y', topMargin + layerIndex * layerSpacing);
      });
    });
  
    if (sigmaInstance) {
      sigmaInstance.refresh();
    }
  } 