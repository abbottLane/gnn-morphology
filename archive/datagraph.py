"""This module contains the DataGraph class.
The DataGraph class is used to encode a sentence as a graph, where each word is a node and each word is 
connected to other words by their dependency relation.

Dependency parses are derived from Spacy's dependency parser. 
"""
import spacy
import torch
NLP = spacy.load("en_core_web_sm")
DEVICE = torch.device("mps")


class DataGraph(object):
    """The DataGraph class is used to encode a sentence as a graph, where each word is a node and each word is 
    connected to other words by their dependency relation.

    Dependency parses are derived from Spacy's dependency parser. 
    """
    def __init__(self, sentence, label):
        """Initialize a DataGraph object.

        Args:
            sentence (str): The sentence to be encoded as a graph.
        """
        self.sentence = sentence
        self.doc = NLP(sentence)
        self.nodes = []
        self.edges = []
        self.node_features = None
        self.label = label
        self._create_graph()
        self._create_node_features()


    def _create_graph(self):
        """Create a graph structure from the sentence.
        """
        # create a node for each word in the sentence
        for token in self.doc:
            self.nodes.append(token.text)

        # create an edge for each dependency relation in the sentence
        for token in self.doc:
            self.edges.append((token.text, token.head.text, token.dep_))
    
    def _create_node_features(self):
        # Extract the set of unique nodes and arcs in the graph
        nodes = set([src for (src, dest, arc) in self.edges])
        arcs = set([arc for (src, dest, arc) in self.edges])

        # Create a dictionary mapping each node and arc to a unique integer index
        node_to_index = {node: i for i, node in enumerate(nodes)}
        arc_to_index = {arc: i for i, arc in enumerate(arcs)}

        # Define the number of nodes and arcs in the graph
        num_nodes = len(nodes)
        num_arcs = len(arcs)

        # Initialize the node features matrix with all zeros
        x = torch.zeros(num_nodes, num_arcs, device=DEVICE)

        # Loop through the tuples in the graph structure
        for (src, dest, arc) in self.edges:
            # Get the index of the source node, destination node, and arc
            src_index = node_to_index[src]
            dest_index = node_to_index[dest]
            arc_index = arc_to_index[arc]

            # Set the corresponding element in the node features matrix to 1
            x[src_index, arc_index] = 1

        # Print the node features matrix
        self.node_features = x

    def __str__(self):
        """Return a string representation of the DataGraph object.
        """
        return f"DataGraph: {self.sentence}"

    def __repr__(self):
        """Return a string representation of the DataGraph object.
        """
        return f"DataGraph: {self.sentence}"
