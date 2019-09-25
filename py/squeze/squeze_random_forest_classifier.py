"""
    SQUEzE
    ======

    This file implements the class RandomForestClassifier,
    that is used to store, train, and predict using a
    random forest model trained with sklearn but modify
    to be persistent.

    It is not an ideal solution but given the time contraints
    it is the fastest way to implement a persistent
    classifer. Future versions of the code should contemplate
    fully deploying our own RandomForestClassifier 
"""
__author__ = "Ignasi Perez-Rafols (iprafols@gmail.com)"
__version__ = "0.1"

import sys

import numpy as np

from squeze.squeze_common_functions import load_pkl, save_pkl

class RandomForestClassifier(object):
    """ The purpose of this class is to create a RandomForestClassifier
        with persistancy. It intends to solve the problem of training
        with sklearn in one environment and not being able to load
        the trained model into another environment.
        It ensures SQUEzE will be able to keep operation even in case of
        updates to sklearn that are not backwards compatible.
        The current

        CLASS: RandomForestClassifier
        PURPOSE: Create a persistent RandomForestClassifier
        """

    def __init__(self, args={}):
        """ Initialize class instance.

            Parameters
            ----------
            args : dict -  Default: {}}
            Options to be passed to the RandomForestClassifier
            """
        self.__args = args
    
        # initialize variables
        self.__num_trees = 0
        self.__trees = []
        self.__num_categories = 0
    
        # auxiliar variables to loop over trees
        self.__children_left = None
        self.__children_right = None
        self.__feature = None
        self.__threshold = None
        self.__proba = None

    def fit(self, X, y):
        """ Create and train models
            
            Parameters
            ----------
            Refer to sklearn.ensemble.RandomForestClassifier.fit
            """
        # load sklearn modules to train the model
        from sklearn.ensemble import RandomForestClassifier as rf_sklearn
        
        # create a RandomForestClassifier
        rf = rf_sklearn(**self.__args)
        
        # train model
        rf.fit(X, y)

        # add persistence
        self.__num_trees = len(rf.estimators_)
        self.__num_categories = np.unique(y).size
        for dt in rf.estimators_:
            tree_sklearn = dt.tree_
            tree = {}
            
            tree["children_left"] = tree_sklearn.children_left
            tree["children_right"] = tree_sklearn.children_right
            tree["feature"] = tree_sklearn.feature
            tree["threshold"] = tree_sklearn.threshold
            value = tree_sklearn.value
            proba = tree_sklearn.value
            for i, p in enumerate(proba):
                proba[i] = p/p.sum()
            tree["proba"] = proba
            
            self.__trees.append(tree)
        
        # discard the sklearn model
        del rf

    def __loadTreeFromForest(self, tree_index):
        """ Loads one tree from the forest file and checks that
            the recursion limit is enough
            
            Parameters
            ----------
            tree_index : int
            Index of the location of the desired tree in self.__trees
            
            """
        self.__children_left = self.__trees[tree_index].get("children_left")
        self.__children_right = self.__trees[tree_index].get("children_right")
        self.__feature = self.__trees[tree_index].get("feature")
        self.__threshold = self.__trees[tree_index].get("threshold")
        self.__tree_proba = self.__trees[tree_index].get("proba")
    
        if len(self.__children_left) > sys.getrecursionlimit():
            sys.setrecursionlimit(int(len(self.__children_left)*1.2))

    def __searchNodes(self, indexs, node_id=0):
        """ Recursively navigates in the tree and calculate the tree response
            
            Parameters
            ----------
            indexs :
            Indexs to question
            """
        # find left child
        left_child_id = self.__children_left[node_id]
        
        # chek if we are on a leave load probabilities and return
        if left_child_id == -1:
            for i in indexs:
                self.__proba[i] = self.__tree_proba[node_id]
            return

        # find right child
        right_child_id = self.__children_right[node_id]

        # find split characteristics
        feature = self.__feature[node_id]
        threshold = self.__threshold[node_id]
        
        # split samples
        left_cond = (self.__X[indexs, feature] <= threshold)
        left_child_indexs = indexs[left_cond]
        right_child_indexs = indexs[~left_cond]

        # navigate the tree
        self.__searchNodes(left_child_indexs, node_id=left_child_id)
        self.__searchNodes(right_child_indexs, node_id=right_child_id)
        
        return
    
    def predict_proba(self, X):
        """ Predict class probabilities for X
            
            Parameters
            ----------
            Refer to sklearn.ensemble.RandomForestClassifier.predic_proba
            """

        output = np.zeros((len(X), self.__num_categories))
        self.__X = X.copy()
        self.__proba = np.zeros((len(X), self.__num_categories))

        for tree_index in np.arange(self.__num_trees):
            self.__loadTreeFromForest(tree_index)
            self.__searchNodes(np.arange(len(X)))
            output += self.__proba

        output /= self.__num_trees
        
        del self.__X
        return output

if __name__ == '__main__':
    pass



