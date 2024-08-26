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
            nodePositions.set(node.id, { x, y });
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
        .attr("r", nodeRadius)
        .attr("cx", d => nodePositions.get(d.id)?.x || 0)
        .attr("cy", d => nodePositions.get(d.id)?.y || 0)
        .attr("fill", d => d3.schemeCategory10[d.layer % 10]);

    console.log("Nodes updated:", node.size());

    node.on("click", handleNodeClick)
        .on("mouseover", handleNodeHover)
        .on("mouseout", handleNodeLeave);
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
    if (selectedNode === d) {
        selectedNode = null;
        resetGraphStyles();
    } else {
        selectedNode = d;
        highlightNode(d);
    }
    updateInfoPanel(d);
}

function handleNodeHover(event, d) {
    console.log("Node hovered:", d);
    if (!selectedNode) {
        highlightNode(d);
    }
}

function handleNodeLeave() {
    if (!selectedNode) {
        resetGraphStyles();
    }
}

function highlightNode(d) {
    const connectedNodes = new Set();
    const connectedLinks = new Set();

    graph.links.forEach(link => {
        if (link.source === d || link.target === d) {
            connectedNodes.add(link.source);
            connectedNodes.add(link.target);
            connectedLinks.add(link);
        }
    });

    svg.selectAll(".node")
        .attr("fill", node => connectedNodes.has(node) ? "blue" : d3.schemeCategory10[node.layer % 10])
        .attr("r", node => connectedNodes.has(node) ? 7 : 5);

    svg.selectAll(".link")
        .attr("stroke", link => connectedLinks.has(link) ? "blue" : "#999")
        .attr("stroke-opacity", link => connectedLinks.has(link) ? 1 : 0.6);
}

function resetGraphStyles() {
    svg.selectAll(".node")
        .attr("fill", d => d3.schemeCategory10[d.layer % 10])
        .attr("r", 5);

    svg.selectAll(".link")
        .attr("stroke", "#999")
        .attr("stroke-opacity", 0.6);
}