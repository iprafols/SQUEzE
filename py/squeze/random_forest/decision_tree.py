"""
    SQUEzE
    ======
    
    This file provides a decision tree used by SQUEzE.
    It contains the following classes:
        - DecisionTree
        - DecisionTreeNode
    This file was based on the code by Jason Browniee
    (but extended for multiple class classifications)
    found here:
    https://machinelearningmastery.com/implement-decision-tree-algorithm-scratch-python/
"""
__author__ = "Ignasi Perez-Rafols (iprafols@gmail.com)"
__version__ = "0.1"

import numpy as np

from squeze.squeze_common_functions import verboseprint

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
        self.__root = None

    def train(self, dataset):
        """ Expand the tree using the given dataset as training set
            
            Parameters
            ----------
            dataset : np.ndarray
            The data the node is responsible for. Must be a structured array
            containing the column 'class'
            """
        self.__root = DecisionTreeNode(dataset)
        self.__root.split(self.__max_depth, self.__min_node_record)

    def print_tree(self, userprint=verboseprint):
        """ Prints the tree
            
            Parameters
            ----------
            userprint : function - Default: verboseprint
            Function to be used for printing
            """
        self.__root.print_node(userprint=userprint)
            
class DecisionTreeNode(object):
    """
        Create a decision tree node
        
        CLASS: DecisionTree
        PURPOSE: Create a decision tree node. This will be the constituents
        of a DecisionTree instance
        """
    def __init__(self, dataset, parent_node=None, depth=0):
        """ Initialize class instance
        
            Parameters
            ----------
            dataset : np.ndarray
            The data the node is responsible for. Must be a structured array
            containing the column 'class'

            parent_node : DecisionTreeNode
            The parent of this node (None for root node)
            
            depth : int
            Tree depth of the node (or the number of nodes that separate
            this node from the root node)
            """
        # keep dataset
        self.__dataset = dataset
        self.__classes = np.unique(dataset["class"])
        
        # set node properties
        self.__parent = parent_node
        self.__depth = depth
        if self.__parent is None:
            self.__initial_classes = self.__classes
        else:
            self.__initial_classes = self.__parent.get_classes()

        # call method 'split' to fill these variables
        # variables for non-terminal nodes
        self.__childs = None
        self.__split_by = None
        self.__split_value = None
        # variables for terminal nodes
        self.__terminal = False
        self.__probs = None

    def get_classes(self):
        """ Return the classes of the initial dataset """
        return self.__initial_classes

    def __gini_index(self, groups, classes):
        """ Compute the Gini index for a list of groups and a list of known
            class values

            Parameters
            ----------
            groups : dict
            A dictionary of groups to be evaluated. Keys are group names
            and values are groups. Each group must contain a N+1 dimensional
            np.array, for an entry with N variables plus the class it belogns
            to (which has to be last).
            
            classes : np.ndarray
            An array with the identifiers of each class. All the identifiers
            should have the same type.

            Returns
            -------
            The Gini index of the split (0.0 for a perfect split)
            """
        # count all samples at split point
        groups = list(groups.values())
        n_instances = np.sum([group.size for group in groups]).astype(float)
        
        # sum weighted Gini index for each group
        gini = 0.0
        for group in groups:
            size = float(group.size)
            # avoid divide by zero
            if size == 0:
                continue
            score = 0.0
            # score the group based on the score for each class
            cls_list = [row["class"] for row in group]
            for cls in self.__classes:
                p = cls_list.count(cls) / size
                score += p * p
            # weight the group score by its relative size
            gini += (1.0 - score) * (size / n_instances)
        return gini

    def __test_split(self, attr, value):
        """ Split the sample into three groups given an index of an
            attribute and a value. The first two groups will be filled with
            data where the attribute is lower or equal (1st group) or higher
            (2nd group) than value. The third group will be filled with data
            where the attribute is NaN.

            Parameters
            ----------
            attr : string
            Name of the property selected for the split. Must be a column
            of self.__dataset

            value : same as the selected property
            Split value

            Returns
            -------
            A dictionary with the three groups
            """
        groups = {}
        groups["leq"] = self.__dataset[np.where((self.__dataset[attr] <= value))]
        groups["gt"] = self.__dataset[np.where((self.__dataset[attr] > value))]
        groups["nan"] = self.__dataset[np.isnan(self.__dataset[attr])]

        return groups

    def __get_split(self):
        """ Select the best split point for a dataset. Basically create different
            test splits using the method __test_split and selects the one with
            better gini score

            Returns
            -------
            A dictionary with the split attribute, the split value, and the
            resulting groups
            """
        best_attr = None
        best_value = 0.0
        best_score = 99999.99
        best_groups = None
        # loop over attributes
        for attr in self.__dataset.dtype.names:
            # ignore if it's the class
            if attr == "class":
                continue
            # loop opver values
            # TODO: consider adding the num_checks as a parameter to the method
            # TODO: increase num_checks to 50 or 100
            num_checks = 10
            # TODO: consider including treatement of categorical variables
            if (np.isnan(np.nanmin(self.__dataset[attr])) or
                np.nanmin(self.__dataset[attr]) == np.nanmax(self.__dataset[attr])):
                continue
            values = np.arange(np.nanmin(self.__dataset[attr]),
                               np.nanmax(self.__dataset[attr]),
                               (np.nanmax(self.__dataset[attr])-np.nanmin(self.__dataset[attr]))/num_checks)
            for value in values:
                groups = self.__test_split(attr, value)
                gini = self.__gini_index(groups, self.__classes)
                if gini < best_score:
                    best_attr, best_value, best_score, best_groups = attr, value, gini, groups
                # if gini of 0 is reached, stop searching
                if best_score == 0.0:
                    break
            if best_score == 0.0:
                break

        return {'attr': best_attr, 'value': best_value, 'groups': best_groups}

    # Create child splits for a node or make terminal
    def split(self, max_depth, min_size):
        """ Split the node recursively. Apply this function on the root node
            to develop the full tree proceding in a depth-first strategy:
            When applied to a non-terminal node, split the sample into three
            nodes and expand each of them. When applied to a terminal node,
            generate the prediction statistics associated to the node.
            
            A node is considered a terminal node if either of the following
            is met:
                1. The node depth equals 'max_depth'
                2. The number of samples in the node is less or equal to
                   'min_size'
                3. All the samples in the node belong to the same class
                4. The sample is already split
            
            Parameters
            ----------
            max_depth : np.ndarray
            The data to split. Must contain the column 'class'. Columns should be
            numerical (except for column 'class').
            """
        if self.__dataset is None:
            return
        if (self.__depth == max_depth or self.__dataset.size <= min_size
            or self.__classes.size <= 1):
            self.__terminal = True
            # compute probabilities
            # TODO: make sure the probabilities are balanced
            self.__probs = [self.__dataset[np.where(self.__dataset["class"] == cls)].size
                            for cls in self.__initial_classes]
            # normalize probabilities
            self.__probs = np.array(self.__probs, dtype=float)/self.__dataset.size
        else:
            split = self.__get_split()
            groups = split["groups"]
            self.__childs = {child: DecisionTreeNode(dataset, parent_node=self,
                                                     depth=self.__depth+1)
                             for child, dataset in groups.items()}
            self.__split_by = split["attr"]
            self.__split_value = split["value"]
        
            for node in list(self.__childs.values()):
                node.split(max_depth, min_size)

        self.__dataset = None


    def print_node(self, userprint=verboseprint):
        """ Prints the information of the node and its children
            
            Parameters
            ----------
            userprint : function - Default: verboseprint
            Function to be used for printing
            """
        # at the beginning of the tree, print the classes
        if self.__parent is None:
            userprint("classes: {}".format(self.__classes))
        
        # node is terminal, print probabilities
        if self.__terminal:
            userprint("{identation}[{probs}]".format(identation=" "*self.__depth,
                                                     probs=self.__probs))
        # node is not terminal, print split information and proceed
        # to print childs
        else:
            # first print this node
            userprint("{identation}[{split_by} <? {split_value}]".format(identation=" "*self.__depth,
                                                                         split_by=self.__split_by,
                                                                         split_value=self.__split_value))
            # now print the chids
            for node in list(self.__childs.values()):
                node.print_node()
