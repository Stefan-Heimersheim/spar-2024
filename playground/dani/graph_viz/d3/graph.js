function initializeSVG() {
    console.log("Initializing SVG...");
    return new Promise(resolve => {
        const checkDimensions = () => {
            const container = document.getElementById('graph-inner-container');
            if (!container) {
                console.error("Graph container not found");
                resolve(null);
                return;
            }
            console.log("Container found:", container);
            container.innerHTML = ''; // Clear existing content

            const rect = container.getBoundingClientRect();
            const width = rect.width;
            const height = rect.height;

            console.log("Container dimensions:", width, height);

            if (width > 0 && height > 0) {
                const svg = d3.select("#graph-inner-container")
                    .append("svg")
                    .attr("width", "100%")
                    .attr("height", "100%");

                // Add a background rectangle
                background = svg.append("rect")
                    .attr("class", "background")
                    .style("position", "absolute")
                    .style("top", "0")
                    .style("left", "0")
                    .style("width", "100%")
                    .style("height", "100%")
                    .style("z-index", "-1");
                background.on("click", handleBackgroundClick);

                console.log("SVG created:", svg.node());
                resolve({ svg, width, height });
            } else {
                requestAnimationFrame(checkDimensions);
            }
        };
        checkDimensions();
    });
}

function getNodesByLayerFeatureIndex(graph) {
    const nodesByLayer = {};
    graph.nodes.forEach(node => {
        if (!nodesByLayer[node.layer]) {
            nodesByLayer[node.layer] = [];
        }
        nodesByLayer[node.layer].push(node);
    });

    Object.keys(nodesByLayer).forEach(layer => {
        nodesByLayer[layer].sort((a, b) => a.feature - b.feature);
    });

    return nodesByLayer;
}

function getNodesByLayerMinimizeCrossings(graph) {
    const nodesByLayer = {};
    graph.nodes.forEach(node => {
        if (!nodesByLayer[node.layer]) {
            nodesByLayer[node.layer] = [];
        }
        nodesByLayer[node.layer].push(node);
    });

    Object.keys(nodesByLayer).forEach(layer => {
        nodesByLayer[layer].sort((a, b) => {
            const aConnections = graph.links.filter(link => link.source === a || link.target === a).length;
            const bConnections = graph.links.filter(link => link.source === b || link.target === b).length;
            return bConnections - aConnections;
        });
    });

    return nodesByLayer;
}

function updateGraph(width, height) {
    console.log("Updating graph");

    const innerContainer = document.getElementById('graph-inner-container');
    const { width: innerWidth, height: innerHeight } = innerContainer.getBoundingClientRect();

    // Update SVG dimensions
    svg.attr("width", innerWidth).attr("height", innerHeight);

    const layerCount = 12; // 0 to 11
    const nodeRadius = 5;
    const margin = {
        top: 20,
        bottom: 20,
        left: 20,
        right: 20
    };

    const sortMethod = document.querySelector('input[name="layer-sort"]:checked').value;
    const nodesByLayer = sortMethod === 'feature-index' 
        ? getNodesByLayerFeatureIndex(graph) 
        : getNodesByLayerMinimizeCrossings(graph);

    const availableWidth = innerWidth - margin.left - margin.right;
    const availableHeight = innerHeight - margin.top - margin.bottom;
    const layerHeight = availableHeight / (layerCount - 1);

    // Create a map of node IDs to their positions
    const nodePositions = new Map();

    // Position nodes
    Object.keys(nodesByLayer).forEach((layer) => {
        const layerIndex = parseInt(layer);
        const nodesInLayer = nodesByLayer[layer];
        const layerY = innerHeight - margin.bottom - layerIndex * layerHeight;
        
        nodesInLayer.forEach((node, nodeIndex) => {
            const nodeSpacing = availableWidth / (nodesInLayer.length + 1);
            const x = margin.left + (nodeIndex + 1) * nodeSpacing;
            const y = layerY;
            nodePositions.set(node.id, { x, y, renderOrder: x });
        });
    });

    // Update links
    const link = svg.selectAll(".link")
    .data(graph.links)
    .join("line")
    .attr("class", "link")
    .attr("stroke-width", d => Math.sqrt(d.similarity) * 2)
    .attr("x1", d => {
        const x = nodePositions.get(d.source)?.x;
        if (x === undefined) console.log("Missing source node:", d.source);
        return x || 0;
    })
    .attr("y1", d => {
        const y = nodePositions.get(d.source)?.y;
        if (y === undefined) console.log("Missing source node:", d.source);
        return y || 0;
    })
    .attr("x2", d => {
        const x = nodePositions.get(d.target)?.x;
        if (x === undefined) console.log("Missing target node:", d.target);
        return x || 0;
    })
    .attr("y2", d => {
        const y = nodePositions.get(d.target)?.y;
        if (y === undefined) console.log("Missing target node:", d.target);
        return y || 0;
    });
    console.log("Links updated:", link.size());

    // Update nodes
    const node = svg.selectAll(".node")
        .data(graph.nodes)
        .join("circle")
        .attr("class", "node")
        .attr("id", d => d.id)
        .attr("r", nodeRadius)
        .attr("cx", d => nodePositions.get(d.id)?.x || 0)
        .attr("cy", d => nodePositions.get(d.id)?.y || 0)
        .attr("fill", d => d3.schemeCategory10[d.layer % 10])
        .sort((a, b) => {
            const orderA = nodePositions.get(a.id)?.renderOrder || 0;
            const orderB = nodePositions.get(b.id)?.renderOrder || 0;
            return orderA - orderB;
        });

    console.log("Nodes updated:", node.size());

    node.on("click", handleNodeClick)
        .on("mouseover", handleNodeHover)
        .on("mouseout", handleNodeLeave);
}

function handleBackgroundClick() {
    selectedNode = null;
    resetGraphStyles();
    updateInfoPanel(null);  // Clear the info panel
}

function drag(simulation) {
    function dragstarted(event) {
        if (!event.active) simulation.alphaTarget(0.3).restart();
        event.subject.fx = event.subject.x;
        event.subject.fy = event.subject.y;
    }

    function dragged(event) {
        event.subject.fx = event.x;
        event.subject.fy = event.y;
    }

    function dragended(event) {
        if (!event.active) simulation.alphaTarget(0);
        event.subject.fx = null;
        event.subject.fy = null;
    }

    return d3.drag()
        .on("start", dragstarted)
        .on("drag", dragged)
        .on("end", dragended);
}

function handleNodeClick(event, d) {
    event.stopPropagation();
    resetGraphStyles();
    unhighlightNode(d);
    highlightNode(d, 'selected', false);
    if (selectedNode === d) {
        selectedNode = null;
        resetGraphStyles();
    } else {
        selectedNode = d;
    }
    updateInfoPanel(d);
}

function handleNodeHover(event, d) {
    highlightNode(d, 'hovered', false);
}

function handleNodeLeave(event, d) {
    unhighlightNode(d);
    resetGraphStyles();
    if (selectedNode) {
        unhighlightNode(selectedNode);
        highlightNode(selectedNode, 'selected', false);
    }
}

function unhighlightNode(d) {
    d.highlighted = null;
    const neighbors = nodeNeighbors.get(d.id);
    const neighborNodeIds = neighbors.nodes;
    
    neighborNodeIds.forEach(nodeId => {
        const node = graph.nodes.find(n => n.id === nodeId);
        if (node.highlighted) {
            unhighlightNode(node);
        }
    });

}

function highlightNode(d, highlightType, distant = false) {
    if (highlightType === d.highlighted) return;
    d.highlighted = highlightType;
    const neighbors = nodeNeighbors.get(d.id);
    const neighborNodeIds = neighbors.nodes;
    const connectedNodes = new Set([d]);
    
    // Add actual node objects to connectedNodes
    neighborNodeIds.forEach(nodeId => {
        const node = graph.nodes.find(n => n.id === nodeId);
        if (node) connectedNodes.add(node);
    });

    const connectedLinks = neighbors.edges;

    const nodeClass = distant ? `${highlightType}-distant-node` : `${highlightType}-node`;
    const edgeClass = distant ? `${highlightType}-distant-edge` : `${highlightType}-edge`;

    nodeSvg = svg.select(`.node[id="${d.id}"]`)
        .classed(nodeClass, true);

    svg.selectAll(".link")
        .classed(edgeClass, link => connectedLinks.has(link))

    neighborNodeIds.forEach(nodeId => {
        const node = graph.nodes.find(n => n.id === nodeId);
        if (node && !node.highlighted) {
            highlightNode(node, highlightType, true);
        }
    });
}

function resetGraphStyles() {
    svg.selectAll(".node")
        .attr("class", "node");

    svg.selectAll(".link")
        .attr("class", "link");
}