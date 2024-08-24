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
    const nodeSize = 5;
    nodes.forEach((node) => {
      const { x, y } = camera.framedGraphToViewport(node);
      context.fillStyle = node.color;
      context.beginPath();
      context.arc(x, y, nodeSize, 0, Math.PI * 2);
      context.fill();

      if (node.highlighted || node.isNeighbor) {
        const tooltip = createNodeTooltip(node.key);
        const tooltipWidth = context.measureText(tooltip).width + 10;
        const tooltipHeight = 20;

        context.fillStyle = "rgba(0, 0, 0, 0.8)";
        context.fillRect(x - tooltipWidth / 2, y - nodeSize - tooltipHeight - 5, tooltipWidth, tooltipHeight);
        context.fillStyle = "#ffffff";
        context.font = "12px Arial";
        context.textAlign = "center";
        context.textBaseline = "middle";
        context.fillText(tooltip, x, y - nodeSize - tooltipHeight / 2 - 5);
      }
    });
  }
  
  function createNodeTooltip(nodeId) {
    const [layer, feature] = nodeId.split('_');
    return `Layer ${layer}, Feature ${feature}`;
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
  
    return { connectedNodes, connectedEdges };
  }
  
  function handleNodeHover(event) {
    const hoveredNodeId = event.node;
    const { connectedNodes, connectedEdges } = getConnectedElements(hoveredNodeId);

    graph.forEachNode((node) => {
      const isConnected = connectedNodes.has(node);
      const isHovered = node === hoveredNodeId;
      
      graph.setNodeAttribute(node, 'color', isConnected ? highlightColor : defaultNodeColor);
      graph.setNodeAttribute(node, 'isNeighbor', isConnected && !isHovered);
      
      if (isConnected) {
        const tooltip = createNodeTooltip(node);
        graph.setNodeAttribute(node, 'tooltip', tooltip);
      } else {
        graph.removeNodeAttribute(node, 'tooltip');
      }
    });

    graph.forEachEdge((edge) => {
      graph.setEdgeAttribute(edge, 'color', connectedEdges.has(edge) ? highlightColor : defaultEdgeColor);
    });

    updateInfoPanel(hoveredNodeId);
    sigmaInstance.refresh();
  }
  
  function handleNodeLeave() {
    graph.forEachNode((node) => {
      graph.setNodeAttribute(node, 'color', defaultNodeColor);
      graph.setNodeAttribute(node, 'isNeighbor', false);
    });

    graph.forEachEdge((edge) => {
      graph.setEdgeAttribute(edge, 'color', defaultEdgeColor);
    });

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