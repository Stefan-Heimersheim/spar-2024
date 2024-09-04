const communityColorScale = d3.scaleOrdinal(d3.schemeCategory10);

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

function getNodesByLayerMaximizeVerticalLines(graph) {
    const nodesByLayer = {};
    // Group nodes by layer
    graph.nodes.forEach(node => {
        if (!nodesByLayer[node.layer]) {
            nodesByLayer[node.layer] = [];
        }
        nodesByLayer[node.layer].push(node);
    });

    // Identify vertical chains
    const verticalChains = identifyVerticalChains(graph, nodesByLayer);
    console.log("Vertical chains:", verticalChains);

    // Assign positions to chains
    const nodePositions = assignPositionsToChains(verticalChains, nodesByLayer);
    console.log("Node positions:", nodePositions);

    // Handle forks and branches
    handleForksAndBranches(graph, nodePositions, nodesByLayer);
    console.log("Node positions after forks and branches:", nodePositions);
    // Optimize remaining node positions
    optimizeRemainingPositions(nodePositions, nodesByLayer);

    // Sort nodes within each layer based on their assigned positions
    Object.keys(nodesByLayer).forEach(layer => {
        nodesByLayer[layer].sort((a, b) => nodePositions[a.id] - nodePositions[b.id]);
    });

    return nodesByLayer;
}

function identifyVerticalChains(graph, nodesByLayer) {
    const chains = [];
    const visited = new Set();

    function dfs(nodeId, currentChain) {
        visited.add(nodeId);
        currentChain.push(nodeId);

        const neighbors = graph.links
            .filter(link => link.source === nodeId || link.target === nodeId)
            .map(link => link.source === nodeId ? link.target : link.source);

        const nextLayerNeighbors = neighbors.filter(neighbor => {
            const [neighborLayer] = neighbor.split('_');
            const [currentLayer] = nodeId.split('_');
            return parseInt(neighborLayer) === parseInt(currentLayer) + 1;
        });

        if (nextLayerNeighbors.length === 1 && !visited.has(nextLayerNeighbors[0])) {
            dfs(nextLayerNeighbors[0], currentChain);
        } else {
            chains.push(currentChain);
        }
    }

    Object.values(nodesByLayer)[0].forEach(node => {
        if (!visited.has(node.id)) {
            dfs(node.id, []);
        }
    });

    return chains;
}

function assignPositionsToChains(verticalChains, nodesByLayer) {
    const nodePositions = {};
    let currentPosition = 0;

    verticalChains.forEach(chain => {
        chain.forEach(nodeId => {
            nodePositions[nodeId] = currentPosition;
        });
        currentPosition++;
    });

    return nodePositions;
}

function handleForksAndBranches(graph, nodePositions, nodesByLayer) {
    graph.links.forEach(link => {
        const sourcePos = nodePositions[link.source];
        const targetPos = nodePositions[link.target];

        if (sourcePos !== undefined && targetPos === undefined) {
            nodePositions[link.target] = sourcePos;
        } else if (sourcePos === undefined && targetPos !== undefined) {
            nodePositions[link.source] = targetPos;
        }
    });
}

function optimizeRemainingPositions(nodePositions, nodesByLayer) {
    Object.values(nodesByLayer).forEach(layerNodes => {
        layerNodes.forEach(node => {
            if (nodePositions[node.id] === undefined) {
                const neighbors = graph.links
                    .filter(link => link.source === node.id || link.target === node.id)
                    .map(link => link.source === node.id ? link.target : link.source);

                const assignedNeighbors = neighbors.filter(neighbor => nodePositions[neighbor] !== undefined);

                if (assignedNeighbors.length > 0) {
                    const avgPosition = assignedNeighbors.reduce((sum, neighbor) => sum + nodePositions[neighbor], 0) / assignedNeighbors.length;
                    nodePositions[node.id] = Math.round(avgPosition);
                } else {
                    nodePositions[node.id] = Object.keys(nodePositions).length;
                }
            }
        });
    });
}

function updateGraph(width, height) {

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
    let nodesByLayer;
    switch (sortMethod) {
        case 'feature-index':
            nodesByLayer = getNodesByLayerFeatureIndex(graph);
            break;
        case 'minimize-crossings':
            nodesByLayer = getNodesByLayerMinimizeCrossings(graph);
            break;
        case 'maximize-vertical-lines':
            nodesByLayer = getNodesByLayerMaximizeVerticalLines(graph);
            break;
        default:
            nodesByLayer = getNodesByLayerMaximizeVerticalLines(graph);
    }

    console.log("Nodes by layer:", nodesByLayer);
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
    .data(graph.links.filter(d => d.similarity > 0))
    .join("line")
    .attr("class", "link")
    .attr("source", d => d.source)
    .attr("target", d => d.target)
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
    })
    .attr("stroke", d => {
        const sourceNode = graph.nodes.find(n => n.id === d.source);
        const targetNode = graph.nodes.find(n => n.id === d.target);
        if (sourceNode && targetNode && sourceNode.community === targetNode.community) {
            return communityColorScale(sourceNode.community);
        }
        return "gray";
    })
    .attr("opacity", 0.5);

    // Update nodes
    const node = svg.selectAll(".node")
        .data(graph.nodes)
        .join("circle")
        .attr("class", "node")
        .attr("id", d => d.id)
        .attr("community", d => d.community || 0)
        .attr("r", nodeRadius)
        .attr("cx", d => nodePositions.get(d.id)?.x || 0)
        .attr("cy", d => nodePositions.get(d.id)?.y || 0)
        .attr("opacity", 0.5)
        .attr("fill", d => {
            const baseColor = communityColorScale(d.community);
            return d3.color(baseColor); // Make the color dim
        })
        .sort((a, b) => {
            const orderA = nodePositions.get(a.id)?.renderOrder || 0;
            const orderB = nodePositions.get(b.id)?.renderOrder || 0;
            return orderA - orderB;
        })
        .raise();

    // Update node styles for hovering and selection
    node.on("mouseover", function(event, d) {
        // d3.select(this).attr("fill", d => {
        //     const baseColor = communityColorScale(d.community);
        //     return d3.color(baseColor); // Increase opacity on hover
        // });
        handleNodeHover(event, d);
    })
    .on("mouseout", function(event, d) {
        // d3.select(this).attr("fill", d => {
        //     const baseColor = communityColorScale(d.community);
        //     return d3.color(baseColor).copy({opacity: 0.3}); // Return to dim color
        // });
        handleNodeLeave(event, d);
    })
    .on("click", handleNodeClick);
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
    showTooltip(d);
}

function highlightHoveredNode(d) {
    if (selectedNode === d || d.highlighted){
        resetGraphStyles();
        unhighlightNode(d);
    }
    highlightNode(d, 'hovered', false);
}

function handleNodeHover(event, d) {
    highlightHoveredNode(d);
    updateInfoPanel(d);
    // Show tooltip
    const tooltip = showTooltip(d);

}

function showTooltip(d) {
    const tooltip = d3.select("body").append("div")
        .attr("class", "tooltip")
        .style("opacity", 0);

    const nodeElement = d3.select(`.node[id="${d.id}"]`)
    const nodeBox = nodeElement.node().getBBox();
    const svgRect = svg.node().getBoundingClientRect();
    const xPosition = svgRect.left + nodeBox.x + nodeBox.width + 10; // 10px to the right of the node
    const yPosition = svgRect.top + nodeBox.y - 10; // 10px above the node

    tooltip.transition()
        .duration(200)
        .style("opacity", .9);

    tooltip.html(`Layer ${d.layer}, Feature ${d.feature}`)
        .style("left", xPosition + "px")
        .style("top", yPosition + "px");

    return tooltip;
}

function handleNodeLeave(event, d) {
    unhighlightNode(d);
    resetGraphStyles();
    if (selectedNode) {
        unhighlightNode(selectedNode);
        highlightNode(selectedNode, 'selected', false);
        updateInfoPanel(selectedNode);
    }
    else {
        updateInfoPanel(null);
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
    // if (highlightType === d.highlighted) return;
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

    connectedLinks.forEach(link => {
        const linkElement = svg.select(`.link[source="${link.source}"][target="${link.target}"]`)
        linkElement.classed(edgeClass, true);
    });

    neighborNodeIds.forEach(nodeId => {
        const node = graph.nodes.find(n => n.id === nodeId);
        if (node && !node.highlighted && node.community === d.community) {
            highlightNode(node, highlightType, true);
        }
    });
}

function resetGraphStyles() {
    svg.selectAll(".node")
        .attr("class", "node");

    svg.selectAll(".link")
        .attr("class", "link");
    d3.select('.tooltip').remove();
}

function highlightOneNode(d, highlightType) {
    const nodeClass = `${highlightType}-node`;
    console.log("nodeClass", nodeClass);
    nodeSvg = svg.select(`.node[id="${d.id}"]`)
        .attr("class", "node")
        .classed(nodeClass, true);
}