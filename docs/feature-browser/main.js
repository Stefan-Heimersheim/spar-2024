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

        loadSampleList().then(() => {
            loadSelectedSample();
        });
    });
}

function loadSampleList() {
    // Hardcoded list of sample files
    const sampleFiles = [
        'abstract.json',
        'ediz_test.json',
        'how_are_you.json',
        'jaccard_leiden_modularity_threshold_0.1_masked_single_0.json',
        'jaccard_leiden_modularity_threshold_0.1_masked_single_12.json',
        'jaccard_leiden_modularity_threshold_0.1_masked_single_23.json',
        'jaccard_leiden_modularity_threshold_0.1_masked_single_32.json',
        'jaccard_leiden_modularity_threshold_0.1_masked_single_54.json',
        'jaccard_leiden_modularity_threshold_0.1_size_11_657.json',
        'jaccard_leiden_modularity_threshold_0.1_size_18_201.json',
        'jaccard_leiden_modularity_threshold_0.1_size_6_1221.json',
        'jaccard_leiden_modularity_threshold_0.1_size_6_1344.json',
        'jaccard_leiden_modularity_threshold_0.1_size_7_1188.json',
        'jaccard_leiden_modularity_threshold_0.1_size_9_787.json',
        'klara_and_the_sun.json',
        'mutual_information_leiden_modularity_threshold_0.1_masked_single_10.json',
        'mutual_information_leiden_modularity_threshold_0.1_masked_single_15.json',
        'mutual_information_leiden_modularity_threshold_0.1_masked_single_21.json',
        'mutual_information_leiden_modularity_threshold_0.1_masked_single_25.json',
        'mutual_information_leiden_modularity_threshold_0.1_masked_single_4.json',
        'mutual_information_leiden_modularity_threshold_0.1_size_12_3.json',
        'mutual_information_leiden_modularity_threshold_0.1_size_5_11.json',
        'mutual_information_leiden_modularity_threshold_0.1_size_5_9.json',
        'mutual_information_leiden_modularity_threshold_0.1_size_6_6.json',
        'mutual_information_leiden_modularity_threshold_0.1_size_9_5.json',
        'necessity_leiden_modularity_threshold_0.1_masked_single_0.json',
        'necessity_leiden_modularity_threshold_0.1_masked_single_13.json',
        'necessity_leiden_modularity_threshold_0.1_masked_single_20.json',
        'necessity_leiden_modularity_threshold_0.1_masked_single_25.json',
        'necessity_leiden_modularity_threshold_0.1_masked_single_8.json',
        'necessity_leiden_modularity_threshold_0.1_size_10_60.json',
        'necessity_leiden_modularity_threshold_0.1_size_13_50.json',
        'necessity_leiden_modularity_threshold_0.1_size_16_42.json',
        'necessity_leiden_modularity_threshold_0.1_size_5_76.json',
        'necessity_leiden_modularity_threshold_0.1_size_8_64.json',
        'pearson_leiden_modularity_threshold_0.1_masked_single_15.json',
        'pearson_leiden_modularity_threshold_0.1_masked_single_2.json',
        'pearson_leiden_modularity_threshold_0.1_masked_single_24.json',
        'pearson_leiden_modularity_threshold_0.1_masked_single_31.json',
        'pearson_leiden_modularity_threshold_0.1_masked_single_8.json',
        'pearson_leiden_modularity_threshold_0.1_masked_size_5_1419.json',
        'pearson_leiden_modularity_threshold_0.1_masked_size_5_1427.json',
        'pearson_leiden_modularity_threshold_0.1_masked_size_5_1436.json',
        'pearson_leiden_modularity_threshold_0.1_masked_size_5_1441.json',
        'pearson_leiden_modularity_threshold_0.1_masked_size_5_1450.json',
        'pearson_leiden_modularity_threshold_0.1_size_12_94.json',
        'pearson_leiden_modularity_threshold_0.1_size_6_213.json',
        'pearson_leiden_modularity_threshold_0.1_size_7_205.json',
        'pearson_leiden_modularity_threshold_0.1_size_8_196.json',
        'pearson_leiden_modularity_threshold_0.1_size_9_192.json',
        'sufficiency_leiden_modularity_threshold_0.1_masked_single_11.json',
        'sufficiency_leiden_modularity_threshold_0.1_masked_single_18.json',
        'sufficiency_leiden_modularity_threshold_0.1_masked_single_2.json',
        'sufficiency_leiden_modularity_threshold_0.1_masked_single_25.json',
        'sufficiency_leiden_modularity_threshold_0.1_masked_single_35.json',
        'sufficiency_leiden_modularity_threshold_0.1_size_11_72.json',
        'sufficiency_leiden_modularity_threshold_0.1_size_13_65.json',
        'sufficiency_leiden_modularity_threshold_0.1_size_14_61.json',
        'sufficiency_leiden_modularity_threshold_0.1_size_5_90.json',
        'sufficiency_leiden_modularity_threshold_0.1_size_9_79.json',
        // Add more sample files as needed
    ];
    

    const sampleSelect = document.getElementById('sample-select');
    sampleFiles.forEach(file => {
        const option = document.createElement('option');
        option.value = file;
        option.textContent = file.replace('.json', '');
        sampleSelect.appendChild(option);
    });
    sampleSelect.addEventListener('change', loadSelectedSample);

    return Promise.resolve(); // Return a resolved promise to maintain the chain
}

function loadSelectedSample() {
    const sampleSelect = document.getElementById('sample-select');
    const selectedSample = sampleSelect.value;
    
    fetch(`./graph_samples/${selectedSample}`)
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

            // Update prompt
            const promptElement = document.getElementById('prompt-text');
            if (data.graph && data.graph.prompt) {
                promptElement.textContent = data.graph.prompt;
            } else {
                promptElement.textContent = "No prompt available.";
            }

            // Add graph description to the info panel
            const graphDescriptionElement = document.getElementById('graph-description');
            if (data.graph && data.graph.description) {
                graphDescriptionElement.textContent = data.graph.description;
            } else {
                graphDescriptionElement.textContent = "No graph description available.";
            }

            const { width, height } = svg.node().getBoundingClientRect();
            updateGraph(width, height);
        })
        .catch(error => console.error("Error loading data:", error));
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

        // Add event listener for collapsible settings
        const collapsible = document.querySelector('.collapsible h3');
        collapsible.addEventListener('click', () => {
            collapsible.parentElement.classList.toggle('active');
            const arrow = collapsible.querySelector('.arrow');
            arrow.textContent = collapsible.parentElement.classList.contains('active') ? '▼' : '▶';
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