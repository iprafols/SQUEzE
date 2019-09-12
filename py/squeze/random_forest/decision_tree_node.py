"""
    SQUEzE
    ======
    
    This file provides the implementation of the decision tree node
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

class DecisionTreeNode(object):
    """
        Create a decision tree node
        
        CLASS: DecisionTree
        PURPOSE: Create a decision tree node. This will be the constituents
        of a DecisionTree instance
        """
    def __init__(self, dataset, initial_classes, name="root", parent_node="", depth=0):
        """ Initialize class instance
        
            Parameters
            ----------
            dataset : np.ndarray
            The data the node is responsible for. Must be a structured array
            containing the column 'class'
            
            initial_classes : np.ndarray
            An array with the possible classes. Must be of the same type as
            dataset["class"] and must contain at least all the item in
            dataset["class"].

            name : str - Default: "root"
            The name of the node
            
            parent_node : str - Default: ""
            The name of the parent node ("" for root node)
            
            depth : int
            Tree depth of the node (or the number of nodes that separate
            this node from the root node)
            """
        # keep dataset
        self.__dataset = dataset
        self.__classes = np.unique(dataset["class"])
        
        # set node properties
        self.parent = parent_node
        self.name = name
        self.__depth = depth
        self.__initial_classes = initial_classes
        
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
    
    def __fast_gini_index(self, attr, value, ):
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
        # sum weighted Gini index for each group (leq, gt, nan)
        pos_leq = np.where((self.__dataset[attr] <= value))
        pos_gt = np.where((self.__dataset[attr] > value))
        pos_nan = np.where(np.isnan(self.__dataset[attr]))
        # TODO: this bit might still be optimized, specially the count part
        cls_list_leq = list(self.__dataset["class"][pos_leq])
        cls_list_gt = list(self.__dataset["class"][pos_gt])
        cls_list_nan = list(self.__dataset["class"][pos_nan])
        size_leq = float(pos_leq[0].size)
        size_gt = float(pos_gt[0].size)
        size_nan = float(pos_nan[0].size)
        score_leq = 0.0
        score_gt = 0.0
        score_nan = 0.0
        # score the group based on the score for each class
        for cls in self.__classes:
            if not size_leq == 0:
                aux = cls_list_leq.count(cls) / size_leq
                score_leq += aux * aux
            if not size_gt == 0:
                aux = cls_list_gt.count(cls) / size_gt
                score_gt += aux * aux
            if not size_nan == 0:
                aux = cls_list_nan.count(cls) / size_nan
                score_nan += aux * aux
        # weight the group score by its relative size
        gini = (1.0 - score_leq) * size_leq
        gini += (1.0 - score_gt) * size_gt
        gini += (1.0 - score_nan) * size_nan
        gini /= float(self.__dataset.size)
        
        return gini
                
    def __fast_gini_index2(self, attr, value, ):
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
        # sum weighted Gini index for each group (leq, gt, nan)
        cls_leq = self.__dataset["class"][np.where((self.__dataset[attr] <= value))]
        cls_gt = self.__dataset["class"][np.where((self.__dataset[attr] > value))]
        cls_nan = self.__dataset["class"][np.where(np.isnan(self.__dataset[attr]))]
        size_leq = float(cls_leq.size)
        size_gt = float(cls_gt.size)
        size_nan = float(cls_nan.size)
        score_leq = 0.0
        score_gt = 0.0
        score_nan = 0.0
        # score the group based on the score for each class
        for cls in self.__classes:
            if not size_leq == 0:
                aux = np.count_nonzero(cls_leq == cls) / size_leq
                score_leq += aux * aux
            if not size_gt == 0:
                aux = np.count_nonzero(cls_gt == cls) / size_gt
                score_gt += aux * aux
            if not size_nan == 0:
                aux = np.count_nonzero(cls_nan == cls) / size_nan
                score_nan += aux * aux
        # weight the group score by its relative size
        gini = (1.0 - score_leq) * size_leq
        gini += (1.0 - score_gt) * size_gt
        gini += (1.0 - score_nan) * size_nan
        gini /= float(self.__dataset.size)
        
        return gini

    def __fast_gini_index3(self, attr, value, ):
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
        # sum weighted Gini index for each group (leq, gt, nan)
        cls_list_leq = self.__dataset["class"][np.where((self.__dataset[attr] <= value))]
        cls_list_gt = self.__dataset["class"][np.where((self.__dataset[attr] > value))]
        cls_list_nan = self.__dataset["class"][np.where(np.isnan(self.__dataset[attr]))]
        size_leq = float(cls_list_leq.size)
        size_gt = float(cls_list_gt.size)
        size_nan = float(cls_list_nan.size)
        # score the group based on the score for each class
        gini = 0.0
        if not size_leq == 0:
            aux = np.array([np.count_nonzero(cls_list_leq == cls)
                            for cls in self.__classes]) / size_leq
            gini += (1.0 - (aux*aux).sum()) * size_leq
        if not size_gt == 0:
            aux = np.array([np.count_nonzero(cls_list_gt == cls)
                            for cls in self.__classes]) / size_gt
            gini += (1.0 - (aux*aux).sum()) * size_gt
        if not size_nan == 0:
            aux = np.array([np.count_nonzero(cls_list_nan == cls)
                            for cls in self.__classes]) / size_nan
            gini += (1.0 - (aux*aux).sum()) * size_nan
        gini /= float(self.__dataset.size)
                
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
        """best_groups = None"""
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
            nanmin = np.nanmin(self.__dataset[attr])
            nanmax = np.nanmax(self.__dataset[attr])
            if (np.isnan(nanmin) or nanmin == nanmax):
                continue
            values = np.arange(nanmin, nanmax, (nanmax-nanmin)/num_checks)
            for value in values:
                # The following is equivalent to running these two commands
                # groups = self.__test_split(attr, value)
                # gini = self.__gini_index(groups, self.__classes)
                gini = self.__fast_gini_index2(attr, value)
                if gini < best_score:
                    # This needs to be replaced if the two commands above are run
                    # best_attr, best_value, best_score, best_groups = attr, value, gini, groups
                    best_attr, best_value, best_score = attr, value, gini
                
                # if gini of 0 is reached, stop searching
                if best_score == 0.0:
                    break
            if best_score == 0.0:
                break
        
        best_groups = self.__test_split(best_attr, best_value)
        return {'attr': best_attr, 'value': best_value, 'groups': best_groups}

    # Create child splits for a node or make terminal
    def split(self, max_depth, min_size):
        """ Split the node. When applied to a non-terminal node, split the
            sample into three nodes and expand each of them. When applied to a
            terminal node, generate the prediction statistics associated to
            the node.
            
            A node is considered a terminal node if either of the following
            is met:
                1. The node depth equals 'max_depth'
                2. The number of samples in the node is less or equal to
                   'min_size'
                3. All the samples in the node belong to the same class
                4. The sample is already split
            
            Parameters
            ----------
            max_depth : int
            Maximum depth of the tree
            
            min_size : int
            Nodes with more samples than min_size are succeptible to be
            split
            
            Returns
            -------
            An empty list for terminal nodes and a list with the childs
            otherwise
            """
        if self.__dataset is None:
            return []
        if (self.__depth == max_depth or self.__dataset.size <= min_size
            or self.__classes.size <= 1):
            self.__terminal = True
            # compute probabilities
            # TODO: make sure the probabilities are balanced
            self.__probs = [self.__dataset[np.where(self.__dataset["class"] == cls)].size
                            for cls in self.__initial_classes]
            # normalize probabilities
            self.__probs = np.array(self.__probs, dtype=float)/self.__dataset.size
            
            # free memory and return
            self.__dataset = None
            return []
        else:
            split = self.__get_split()
            groups = split.get("groups")
            # TODO: remove old stuff
            """self.__childs = {child: DecisionTreeNode(dataset, self.__initial_classes,
                                                     name="{}_{}".format(self.name, child),
                                                     parent_node="{}".format(self.name),
                                                     depth=self.__depth+1)
                             for child, dataset in groups.items()}"""
            self.__childs = {child: "{}_{}".format(self.name, child) for child in groups}
            self.__split_by = split.get("attr")
            self.__split_value = split.get("value")

            # free memory and return
            self.__dataset = None
            split_nodes = [DecisionTreeNode(dataset, self.__initial_classes,
                                            name="{}_{}".format(self.name, child),
                                            parent_node="{}".format(self.name),
                                            depth=self.__depth+1)
                           for child, dataset in groups.items()]
            return split_nodes
            # TODO: remove old stuff
            """return list(self.__childs.values())"""

    def predict(self, row):
        """ Classify a single row. The classification is made navigating the
            trained tree
            
            Parameters
            ----------
            row : np.ndarray
            A structured array row containing the columns used for training
            
            Returns
            -------
            If the node is a terminal node, a tuple with the classes and the
            probabilities of each class. Otherwise, the next node to check
            """
        if self.__terminal == True:
            return (self.__initial_classes, self.__probs)
        else:
            if np.isnan(row[self.__split_by]):
                # TODO: remove old stuff
                """return (self.__childs["nan"].name,)"""
                return (self.__childs["nan"],)
            elif row[self.__split_by] <= self.__split_value:
                # TODO: remove old stuff
                """return (self.__childs["leq"].name,)"""
                return (self.__childs["leq"],)
            else:
                # TODO: remove old stuff
                """return (self.__childs["gt"].name,)"""
                return (self.__childs["gt"],)

    def print_node(self, nodes, userprint=verboseprint):
        """ Prints the information of the node and its children
            
            Parameters
            ----------
            nodes : dictionary
            A dictionary with the nodes and their names
            
            userprint : function - Default: verboseprint
            Function to be used for printing
            """
        # at the beginning of the tree, print the classes
        if self.name is "root":
            userprint("classes: {}".format(self.__initial_classes))
        
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
            for node_name in list(self.__childs.values()):
                nodes[node_name].print_node(nodes)
