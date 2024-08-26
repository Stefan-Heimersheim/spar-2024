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

function updateGraph(svg, width, height) {
    console.log("Updating graph");

    const layerCount = 12; // Assuming 12 layers for GPT2
    const layerHeight = height / layerCount;
    const nodeRadius = 5;

    // Group nodes by layer
    const nodesByLayer = {};
    graph.nodes.forEach(node => {
        if (!nodesByLayer[node.layer]) {
            nodesByLayer[node.layer] = [];
        }
        nodesByLayer[node.layer].push(node);
    });

    // Position nodes
    Object.keys(nodesByLayer).forEach((layer, layerIndex) => {
        const nodesInLayer = nodesByLayer[layer];
        const layerY = layerIndex * layerHeight + layerHeight / 2;
        
        nodesInLayer.forEach((node, nodeIndex) => {
            const layerWidth = width - 2 * nodeRadius;
            node.x = (nodeIndex / (nodesInLayer.length - 1)) * layerWidth + nodeRadius;
            node.y = layerY;
        });
    });

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

    // Remove the simulation as it's no longer needed
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