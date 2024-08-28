let graph;
let selectedNode = null;
let svg;
let nodeNeighbors = new Map();

function initializeGraph() {
    console.log("Initializing graph...");
    initializeSVG().then(({ svg: svgElement, width, height }) => {
        if (!svgElement) return;
        console.log("Container initialized with dimensions:", width, height);
        svg = svgElement;  // Assign to the global svg variable

        fetch('active_features_2.json')
            .then(response => response.json())
            .then(data => {
                console.log("Data loaded:", data);
                data.links = data.links.filter(link => link.similarity > 0);
                
                // Apply sentence case to node explanations
                data.nodes.forEach(node => {
                    if (node.explanation) {
                        node.explanation = toSentenceCase(node.explanation);
                    }
                });

                graph = data;
                initializeNodeNeighbors();
                console.log("Graph nodes:", graph.nodes.length, "Graph links:", graph.links.length);
                updateGraph(width, height);
            })
            .catch(error => console.error("Error loading data:", error));
    });
}

// Helper function to convert a string to sentence case
function toSentenceCase(str) {
    return str.charAt(0).toUpperCase() + str.slice(1).toLowerCase();
}

function ensureDOMLoaded(callback) {
    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", callback);
    } else {
        callback();
    }
}

ensureDOMLoaded(() => {
    console.log("DOM fully loaded");
    // Wait for the next frame to ensure styles are applied
    requestAnimationFrame(() => {
        console.log("Styles should be applied now");
        initializeGraph();

        // Add event listener for layout change
        document.querySelectorAll('input[name="layer-sort"]').forEach(radio => {
            radio.addEventListener('change', () => {
                const { width, height } = svg.node().getBoundingClientRect();
                updateGraph(width, height);
            });
        });
    });
});

window.addEventListener("resize", () => {
    console.log("Window resized");
    const container = document.getElementById('graph-inner-container');
    const { width, height } = container.getBoundingClientRect();
    if (graph && svg) {
        updateGraph(width, height);
    }
});

function initializeNodeNeighbors() {
    nodeNeighbors = new Map();
    graph.nodes.forEach(node => {
        nodeNeighbors.set(node.id, { nodes: new Set(), edges: new Set() });
    });

    graph.links.forEach(link => {
        const sourceNeighbors = nodeNeighbors.get(link.source);
        const targetNeighbors = nodeNeighbors.get(link.target);

        sourceNeighbors.nodes.add(link.target);
        sourceNeighbors.edges.add(link);

        targetNeighbors.nodes.add(link.source);
        targetNeighbors.edges.add(link);
    });
}