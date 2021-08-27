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
import astropy.io.fits as fits
import numba
from numba import prange, jit

from squeze.common_functions import deserialize

@jit(nopython=True,
     locals=dict(X=numba.types.float64[:, :],
                 children_left=numba.types.int64[:],
                 children_right=numba.types.int64[:],
                 thresholds=numba.types.float64[:],
                 tree_proba=numba.types.float64[:, :, :],
                 proba=numba.types.float64[:, :],
                 indexs=numba.types.int64[:],
                 node_id=numba.types.int64),
     )
def search_nodes(X, children_left, children_right, features, thresholds,
                tree_proba, proba, indexs, node_id):
    """ Recursively navigates in the tree and calculate the tree response
        by updating the values stored in self.__proba

        Parameters
        ----------
        X : array of floats
        The dataset to classify. Each line contains the information for a given
        candidate

        children_left : array of int
        For node i, j=children_left[i] is the position of left tree node. -1
        indicates a leaf node

        children_right : array of int
        For node i, j=children_left[i] is the position of right tree node. -1
        indicates a leaf node

        features : array of int
        For node i, j=features[i] is the index of the feature used to split the
        dataset

        thresholds : array of float
        For node i, y=thresholds[i] is the threshold used to split the
        dataset. Candidates where its feature j is evaluated to less or equal y
        are assigned to the left child node, others are assigned to the right
        child node

        tree_proba : array of float
        Probabilities for the different leafs of the tree and for the different
        classes

        proba : array of float
        Probabilities assigned by the tree to the different candidates

        index : array of int
        Indexs of the candidates being evaluated

        node_id : int
        Current node
        """
    # find left child
    left_child_id = children_left[node_id]

    # chek if we are on a leave load probabilities and return
    if left_child_id == -1:
        for i in prange(indexs.size):
            proba[indexs[i]] = tree_proba[node_id]
        return proba

    # find right child
    right_child_id = children_right[node_id]

    # find split characteristics
    feature = features[node_id]
    threshold = thresholds[node_id]

    # split samples
    left_cond = (X[indexs, feature] <= threshold)
    left_child_indexs = indexs[left_cond]
    right_child_indexs = indexs[~left_cond]

    # navigate the tree
    proba = search_nodes(X, children_left, children_right, features,
                thresholds, tree_proba, proba, left_child_indexs,
                left_child_id)
    proba = search_nodes(X, children_left, children_right, features,
                thresholds, tree_proba, proba, right_child_indexs,
                right_child_id)

    return proba

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

    def __init__(self, **kwargs):
        """ Initialize class instance.

            Parameters
            ----------
            args : dict -  Default: {}}
            Options to be passed to the RandomForestClassifier
            """
        self.__args = kwargs

        # initialize variables
        self.__num_trees = 0
        self.__trees = []
        self.__num_categories = 0
        self.classes_ = []

        # auxiliar variables to loop over trees
        self.__children_left = None
        self.__children_right = None
        self.__feature = None
        self.__threshold = None
        self.__proba = None
        self.__tree_proba = None

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
        self.classes_ = rf.classes_
        for dt in rf.estimators_:
            tree_sklearn = dt.tree_

            tree = {}
            tree["children_left"] = tree_sklearn.children_left.astype(int)
            tree["children_right"] = tree_sklearn.children_right.astype(int)
            tree["feature"] = tree_sklearn.feature.astype(int)
            tree["threshold"] = tree_sklearn.threshold.astype(float)
            value = tree_sklearn.value.astype(float)
            proba = tree_sklearn.value.astype(float)
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
        self.__children_left = self.__trees[tree_index]["children_left"]
        self.__children_right = self.__trees[tree_index]["children_right"]
        self.__feature = self.__trees[tree_index]["feature"]
        self.__threshold = self.__trees[tree_index]["threshold"]
        self.__tree_proba = self.__trees[tree_index]["proba"]

        if len(self.__children_left) > sys.getrecursionlimit():
            sys.setrecursionlimit(int(len(self.__children_left)*1.2))

    def __searchNodes(self, indexs, node_id=0):
        """ Recursively navigates in the tree and calculate the tree response
            by updating the values stored in self.__proba

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
        print(node_id, self.__children_left[node_id], self.__children_right[node_id], len(self.__children_left))
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
        #self.__X = X.copy()
        #self.__proba = np.zeros((len(X), self.__num_categories))

        for tree_index in np.arange(self.__num_trees):
            #self.__loadTreeFromForest(tree_index)
            #self.__searchNodes(np.arange(len(X)))
            #output += self.__proba
            proba = np.zeros((len(X), self.__num_categories))
            children_left = self.__trees[tree_index]["children_left"]
            children_right = self.__trees[tree_index]["children_right"]
            features = self.__trees[tree_index]["feature"]
            thresholds = self.__trees[tree_index]["threshold"]
            tree_proba = self.__trees[tree_index]["proba"]
            indexs = np.arange(X.shape[0], dtype=int)
            """import numba
            print("children_left", children_left.dtype, numba.typeof(children_left))
            print("children_right", children_right.dtype, numba.typeof(children_right))
            print("features", features.dtype, numba.typeof(features))
            print("thresholds", thresholds.dtype, numba.typeof(thresholds))
            print("tree_proba", tree_proba.dtype, numba.typeof(tree_proba))
            print("indexs", indexs.dtype, numba.typeof(indexs))"""
            if len(children_left) > sys.getrecursionlimit():
                sys.setrecursionlimit(int(len(children_left)*1.2))
            search_nodes(X, children_left, children_right, features, thresholds,
                         tree_proba, proba, indexs, 0)
            output += proba
            #raise Exception("No need to continue")

        output /= self.__num_trees

        #del self.__X
        return output

    @classmethod
    def from_json(cls, data):
        """ This function deserializes a json string to correclty build the class.
            It uses the deserialization function of class SimpleSpectrum to reconstruct
            the instances of Spectrum. For this function to work, data should have been
            serialized using the serialization method specified in `save_json` function
            present on `squeze_common_functions.py` """

        # create instance using the constructor
        cls_instance = cls(**data.get("_RandomForestClassifier__args"))

        # now update the instance to the current values
        cls_instance.set_num_trees(data.get("_RandomForestClassifier__num_trees"))
        cls_instance.set_num_categories(data.get("_RandomForestClassifier__num_categories"))
        cls_instance.classes_ = deserialize(data.get("classes_"))

        trees = data.get("_RandomForestClassifier__trees")
        for tree in trees:
            for key, value in tree.items():
                tree[key] = deserialize(value)
        cls_instance.set_trees(trees)

        return cls_instance

    @classmethod
    def from_fits_hdul(cls, hdul, name_prefix, num_trees, num_categories,
                       classes, args={}):
        """ This function parses the RandomForestClassifier from the data
            contained in a fits HDUList. Each HDU in HDUL has to be according
            to the format specified in method to_fits_hdu

            Parameters
            ----------
            hdul : fits.hdu.hdulist.HDUList
            The Header Data Unit Containing the trained classifier

            name_prefix : string
            Prefix of the HDU names (high, low, or all)

            num_trees : int
            Number of trees in the forest

            num_categories : int
            Number of categories to classify

            classes : array
            Classes present in the data

            args : dict -  Default: {}}
            Options to be passed to the RandomForestClassifier

            """
        # create instance using the constructor
        cls_instance = cls(**args)

        # now update the instance to the current values
        cls_instance.set_num_trees(num_trees)
        cls_instance.set_num_categories(num_categories)
        cls_instance.classes_ = classes

        hdus = [hdul["{}{}".format(name_prefix, index)]
                for index in range(num_trees)]
        trees = [{"children_left": hdu.data["children_left"].astype(np.int64),
                  "children_right": hdu.data["children_right"].astype(np.int64),
                  "feature": hdu.data["feature"].astype(np.int64),
                  "threshold": hdu.data["threshold"].astype(np.float64),
                  "proba": hdu.data["proba"].astype(np.float64).reshape(
                    (hdu.data["proba"].shape[0], 1, hdu.data["proba"].shape[1])),
                } for hdu in hdus]
        cls_instance.set_trees(trees)
        
        return cls_instance

    def set_num_trees(self, num_trees):
        """ Set the variable __num_trees. Should only be called from the method from_json"""
        self.__num_trees = num_trees

    def set_num_categories(self, num_categories):
        """ Set the variable __num_categories. Should only be called from the method from_json"""
        self.__num_categories = num_categories

    def set_trees(self, trees):
        """ Set the variable __trees. Should only be called from the method from_json"""
        self.__trees = trees

    def num_trees(self):
        """ Access the number of trees """
        return self.__num_trees

    def num_categories(self):
        """ Access the number of categories """
        return self.__num_categories

    def to_fits_hdu(self, index, name):
        """ Formats tree as a fits Header Data Unit

            Parameters
            ----------
            index : int
            Index of the tree to format

            name : string
            Name of the HDU

            Returns
            -------
            The Header Data Unit
            """
        # create HDU columns
        cols = [fits.Column(name="{}".format(field),
                    array=self.__trees[index].get(field),
                    format=type,
                    )
                for field, type in [("children_left", "K"),
                                    ("children_right", "K"),
                                    ("feature", "K"),
                                    ("threshold", "D"),
                                    ("proba", "{}D".format(self.__num_categories))]
                ]

        # create HDU and return
        return fits.BinTableHDU.from_columns(cols, name=name)

if __name__ == '__main__':
    pass
