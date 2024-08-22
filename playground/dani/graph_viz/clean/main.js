let sigmaInstance;
let graph;

const highlightColor = '#0000FF';
const defaultNodeColor = '#000000';
const defaultEdgeColor = '#CCCCCC';

function initializeSigma() {
  const container = document.getElementById('sigma-container');
  sigmaInstance = new Sigma(graph, container, {
    renderEdgeLabels: false,
    renderNodes: renderCustomNodes,
  });

  sigmaInstance.on('enterNode', handleNodeHover);
  sigmaInstance.on('leaveNode', handleNodeLeave);
}

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