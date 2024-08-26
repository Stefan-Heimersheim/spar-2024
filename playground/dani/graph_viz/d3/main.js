let graph;
let selectedNode = null;

function initializeGraph() {
    console.log("Initializing graph...");
    initializeSVG().then(({ svg, width, height }) => {
        if (!svg) return;
        console.log("Container initialized with dimensions:", width, height);

        fetch('sample_graph.json')
            .then(response => response.json())
            .then(data => {
                console.log("Data loaded:", data);
                graph = data;
                console.log("Graph nodes:", graph.nodes.length, "Graph links:", graph.links.length);
                updateGraph(svg, width, height);
            })
            .catch(error => console.error("Error loading data:", error));
    });
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
    });
});

window.addEventListener("resize", () => {
    console.log("Window resized");
    const { svg, width, height } = initializeSVG();
    if (graph) {
        updateGraph(svg, width, height);
    }
});