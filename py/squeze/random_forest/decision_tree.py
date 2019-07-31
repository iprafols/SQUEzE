"""
    SQUEzE
    ======
    
    This file provides the implementation of the decision tree
    used by SQUEzE.
    This file was based on the code by Jason Browniee
    (but extended for multiple class classifications)
    found here:
    https://machinelearningmastery.com/implement-decision-tree-algorithm-scratch-python/
"""
__author__ = "Ignasi Perez-Rafols (iprafols@gmail.com)"
__version__ = "0.1"

import numpy as np

from squeze.squeze_common_functions import verboseprint
from squeze.random_forest.decision_tree_node import DecisionTreeNode
from squeze.squeze_error import Error

class DecisionTree(object):
    """
        Create a decision tree
        
        CLASS: DecisionTree
        PURPOSE: Create a decision tree to be used by SQUEzE in the context
        of a RandomForest Classifier
        """
    def __init__(self, max_depth, min_node_record):
        """ Initialize class instance
            
            Parameters
            ----------
            max_depth : int
            Maximum depth of the tree
            
            min_node_record : int
            Minimum number of entries a given node is responsible for.
            Once at or below this minimum nodes will not be split.
            """
        self.__max_depth = max_depth
        self.__min_node_record = min_node_record
        
        # call method 'train' to create the tree
        self.__nodes = {}
    
    def __predict(self, row):
        """ Classify a single row. The classification is made navigating the
            trained tree
            
            Parameters
            ----------
            row : np.ndarray
            A structured array row containing the columns used for training
            
            Returns
            -------
            A tuple with the classes and the probabilities for each class
            """
        current_node = self.__nodes.get("root", None)
        prediction = current_node.predict(row)
        while len(prediction) == 1:
            current_node = self.__nodes.get(prediction[0], None)
            if current_node is None:
                raise Error("Error while following tree.")
            prediction = current_node.predict(row)
        return prediction
    
    def get_classes(self):
        """ Return the classes of the initial dataset """
        return self.__nodes.get("root", None).get_classes()
    
    def train(self, dataset):
        """ Expand the tree using the given dataset as training set.
            Split the nodes recursively to develop the full tree
            proceding in a depth-first strategy
            
            Parameters
            ----------
            dataset : np.ndarray
            The data the node is responsible for. Must be a structured array
            containing the column 'class'
            """
        unexpanded_nodes = [DecisionTreeNode(dataset, np.unique(dataset["class"]),
                                             name="root")]
        while len(unexpanded_nodes) > 0:
            node = unexpanded_nodes.pop()
            unexpanded_nodes += node.split(self.__max_depth, self.__min_node_record)
            self.__nodes[node.name] = node
            
    def print_tree(self, userprint=verboseprint):
        """ Prints the tree
            
            Parameters
            ----------
            userprint : function - Default: verboseprint
            Function to be used for printing
            """
        self.__nodes["root"].print_node(userprint=userprint)
            
    def predict_proba(self, dataset, reset_names=False):
        """ Classify the data.
            
            Parameters
            ----------
            dataset : np.ndarray
            The data to classify. It must be a structured array row containing the
            columns used for training (except for the column 'class')
            
            reset_names : bool - Default: False
            If True, returns the array as is, without column names
            
            Returns
            -------
            An array with the prediction for each row in the dataset
            """
        try:
            classes = self.get_classes()
        except AttributeError:
            raise Error("Attempted prediction on untrained tree.")

        if reset_names == False:
            probabilities = np.zeros((dataset.size, classes.size), dtype=float)
        else:
            dtype = [(str(cls), float) for cls in classes]
            probabilities = np.zeros(dataset.size, dtype=dtype)

        for index, row in enumerate(dataset):
            probabilities[index] = tuple(self.__predict(row)[1])

        return probabilities
