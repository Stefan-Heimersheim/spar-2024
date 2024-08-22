let sigmaInstance;
let graph;

const highlightColor = '#0000FF';
const defaultNodeColor = '#000000';
const defaultEdgeColor = '#CCCCCC';

fetch('sample_graph.json')
  .then(response => response.json())
  .then(data => {
    graph = new graphology.Graph();
    initializeGraph(data);
    initializeSigma();
    updateLayout();
  })
  .catch(error => console.error('Error:', error));

window.addEventListener('resize', updateLayout);