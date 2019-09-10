"""
    SQUEzE
    ======
    
    This file provides the implementation of the random forest
    used by SQUEzE.
"""
__author__ = "Ignasi Perez-Rafols (iprafols@gmail.com)"
__version__ = "0.1"

import numpy as np

from squeze.squeze_common_functions import verboseprint
from squeze.random_forest.decision_tree import DecisionTree

class RandomForestClassifier(object):
    """
        Create a random forest classifier
        
        CLASS: RandomForestClassifier
        PURPOSE: Create a random forest classifier to be used by
        SQUEzE. Take the class DecisionTree from squeze.random_forest
        as trees in the forest.
        """
    def __init__(self, num_estimators=1000, max_depth=10, min_node_record=2,
                 random_state=None):
        """ Initialize class instance
            
            Parameters
            ----------
            num_estimators : int - Default: 1000
            Number of trees in the forest
            
            max_depth : int - Default: 10
            Maximum depth of the tree
            
            min_node_record : int - Default: 2
            Minimum number of entries a given node is responsible for.
            Once at or below this minimum nodes will not be split.
            
            random_state : int, RandomState instance or None - Default: None
            If int, random_state is the seed used by the random number generator;
            If RandomState instance, random_state is the random number generator;
            If None, the random number generator is the RandomState instance used
            by np.random.
            """
        self.__max_depth = max_depth
        self.__min_node_record = min_node_record
        self.__num_estimators = num_estimators
        self.__random_state = random_state
        
        # create untrained trees
        self.__trees = [DecisionTree(max_depth, min_node_record) for item in range(num_estimators)]

    def get_classes(self):
        """ Return the classes of the initial dataset """
        return self.__trees[0].get_classes()
    
    def train(self, dataset):
        """ Train all the trees in the forest using bootstrap samples of the
            dataset
            
            Parameters
            ----------
            dataset : np.ndarray
            The data the node is responsible for. Must be a structured array
            containing the column 'class'
            """

        # reinitialize random number generator
        np.random.seed(self.__random_state)

        # draw bootstrap samples to train each tree
        for i in range(self.__num_estimators):
            selected_rows = np.random.randint(0, dataset.size, size=dataset.size)
            self.__trees[i].train(dataset[selected_rows])

    def predict_proba(self, dataset):
        """ Classify the data.
            
            Parameters
            ----------
            dataset : np.ndarray
            The data to classify. It must be a structured array row containing the
            columns used for training (except for the column 'class')
            
            Returns
            -------
            An array with the prediction for each row in the dataset
            """
        try:
            classes = self.__trees[0].get_classes()
        except AttributeError:
            raise Error("Attempted prediction on untrained tree.")
        
        # compute probabilities for each of the trees
        """aux = np.zeros((self.__num_estimators, dataset.size), dtype=float)
        for index, tree in enumerate(self.__trees):
            aux[index] = tree.predict_proba(dataset)
        """
        probs = np.array([tree.predict_proba(dataset) for tree in self.__trees])
        
        # assign the classes to each value
        probabilities = np.nanmean(probs, axis=0)
        dtype = [(str(cls), float) for cls in classes]
        probabilities.dtype = dtype

        return probabilities

