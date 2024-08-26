function initializeSVG() {
    console.log("Initializing SVG...");
    return new Promise(resolve => {
        const checkDimensions = () => {
            const container = document.getElementById('graph-container');
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
                const svg = d3.select("#graph-container")
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

    const layerCount = 12;
    const nodeRadius = 5;
    const extraMargin = 20;

    const sortMethod = document.querySelector('input[name="layer-sort"]:checked').value;
    const nodesByLayer = sortMethod === 'feature-index' 
        ? getNodesByLayerFeatureIndex(graph) 
        : getNodesByLayerMinimizeCrossings(graph);

    const availableWidth = width - 2 * extraMargin;
    const availableHeight = height - 2 * extraMargin;
    const layerHeight = availableHeight / (layerCount - 1);

    Object.keys(nodesByLayer).forEach((layer, layerIndex) => {
        const nodesInLayer = nodesByLayer[layer];
        const layerY = height - extraMargin - layerIndex * layerHeight;
        
        nodesInLayer.forEach((node, nodeIndex) => {
            const nodeSpacing = availableWidth / (nodesInLayer.length + 1);
            node.x = extraMargin + (nodeIndex + 1) * nodeSpacing;
            node.y = layerY;
        });
    });

    // Update links
    const link = svg.selectAll(".link")
        .data(graph.links)
        .join("line")
        .attr("class", "link")
        .attr("stroke-width", d => Math.sqrt(d.similarity) * 2)
        .attr("x1", d => d.source.x)
        .attr("y1", d => d.source.y)
        .attr("x2", d => d.target.x)
        .attr("y2", d => d.target.y);

    console.log("Links updated:", link.size());

    // Update nodes
    const node = svg.selectAll(".node")
        .data(graph.nodes)
        .join("circle")
        .attr("class", "node")
        .attr("r", nodeRadius)
        .attr("cx", d => d.x)
        .attr("cy", d => d.y)
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