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

import sys

import numpy as np
import numba
from numba import prange, jit

from squeze.utils import deserialize

# extra imports for training model
SKLEARN_ERROR = None
try:
    from sklearn.ensemble import RandomForestClassifier as rf_sklearn
except ImportError as error:
    SKLEARN_ERROR = error
# load sklearn modules to train the model


@jit(
    nopython=True,
    locals={
        "X": numba.types.float64[:, :],
        "children_left": numba.types.int64[:],
        "children_right": numba.types.int64[:],
        "thresholds": numba.types.float64[:],
        "tree_proba": numba.types.float64[:, :, :],
        "proba": numba.types.float64[:, :],
        "indexs": numba.types.int64[:],
        "node_id": numba.types.int64},
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
        # pylint: disable=not-an-iterable
        # prange is the numba equivalent to range
        for index1 in prange(indexs.size):
            proba[indexs[index1]] = tree_proba[node_id]
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
    proba = search_nodes(X, children_left, children_right, features, thresholds,
                         tree_proba, proba, left_child_indexs, left_child_id)
    proba = search_nodes(X, children_left, children_right, features, thresholds,
                         tree_proba, proba, right_child_indexs, right_child_id)

    return proba


class RandomForestClassifier:
    """ The purpose of this class is to create a RandomForestClassifier
    with persistancy. It intends to solve the problem of training
    with sklearn in one environment and not being able to load
    the trained model into another environment.
    It ensures SQUEzE will be able to keep operation even in case of
    updates to sklearn that are not backwards compatible.

    CLASS: RandomForestClassifier
    PURPOSE: Create a persistent RandomForestClassifier
    """

    def __init__(self, **kwargs):
        """ Initialize class instance.

        Arguments
        ---------
        kwargs: dict
        Options to be passed to the RandomForestClassifier
        """
        self.args = kwargs

        # initialize variables
        self.num_trees = 0
        self.trees = []
        self.num_categories = 0
        self.classes = []
        self.feature_importances = []

    def fit(self, X, y):
        """ Create and train models

        Arguments
        ---------
        Refer to sklearn.ensemble.RandomForestClassifier.fit
        """
        if SKLEARN_ERROR is not None:
            raise SKLEARN_ERROR

        # create a RandomForestClassifier
        random_forest = rf_sklearn(**self.args)

        # train model
        random_forest.fit(X, y)

        # add persistence
        self.num_trees = len(random_forest.estimators_)
        self.num_categories = np.unique(y).size
        self.classes = random_forest.classes_
        self.feature_importances = random_forest.feature_importances_
        for decision_tree in random_forest.estimators_:
            tree_sklearn = decision_tree.tree_

            tree = {}
            tree["feature_importance"] = decision_tree.feature_importances_.astype(float)
            tree["children_left"] = tree_sklearn.children_left.astype(int)
            tree["children_right"] = tree_sklearn.children_right.astype(int)
            tree["feature"] = tree_sklearn.feature.astype(int)
            tree["threshold"] = tree_sklearn.threshold.astype(float)
            proba = tree_sklearn.value.astype(float)
            for index, prob in enumerate(proba):
                proba[index] = prob / prob.sum()
            tree["proba"] = proba

            self.trees.append(tree)

        # discard the sklearn model
        del random_forest

    @classmethod
    def from_json(cls, data):
        """ This function deserializes a json string to correclty build the class.
        It uses the deserialization function of class SimpleSpectrum to reconstruct
        the instances of Spectrum. For this function to work, data should have been
        serialized using the serialization method specified in `save_json` function
        present on `utils.py`
        """

        # create instance using the constructor
        cls_instance = cls(**data.get("args"))

        # now update the instance to the current values
        cls_instance.num_trees = data.get("num_trees")
        cls_instance.num_categories = data.get("num_categories")
        cls_instance.classes = deserialize(data.get("classes"))
        # DEPRECATED: older models might not have feature importance
        # at some point this will be removed
        feature_importances = data.get("feature_importances")
        if feature_importances is None:
            cls_instance.feature_importances = feature_importances
        else:
            cls_instance.feature_importances = deserialize(feature_importances)

        trees = data.get("trees")
        for tree in trees:
            for key, value in tree.items():
                tree[key] = deserialize(value)
        cls_instance.trees = trees

        return cls_instance

    @classmethod
    def from_fits_hdul(cls, hdul, name_prefix, info_name, args=None):
        """ This function parses the RandomForestClassifier from the data
        contained in a fits HDUList. Each HDU in HDUL has to be according
        to the format specified in method to_fits_hdu

        Arguments
        ---------
        hdul : fitsio.fitslib.FITS
        The Header Data Unit List containing the trained classifier

        name_prefix : string
        Prefix of the HDU names (high, low, or all)

        name_prefix : string
        Name of the info HDU (HIGHINFO, LOWINFO, or ALLINFO)

        args : dict -  Default: {}}
        Options to be passed to the RandomForestClassifier

        """
        if args is None:
            args = {}

        # create instance using the constructor
        cls_instance = cls(**args)

        # now update the instance to the current values
        header = hdul[info_name].read_header()
        cls_instance.num_trees = header["N_TREES"]
        cls_instance.num_categories = header["N_CAT"]
        cls_instance.classes = hdul[info_name]["CLASSES"][:].astype(np.float64)

        hdus = [
            hdul[f"{name_prefix}{index}"] for index in range(header["N_TREES"])
        ]
        trees = [{
            "children_left": hdu["children_left"][:].astype(np.int64),
            "children_right": hdu["children_right"][:].astype(np.int64),
            "feature": hdu["feature"][:].astype(np.int64),
            "threshold": hdu["threshold"][:].astype(np.float64),
            "proba": hdu["proba"][:].astype(np.float64),
        } for hdu in hdus]
        cls_instance.trees = trees
        return cls_instance

    def predict_proba(self, X):
        """ Predict class probabilities for X

        Arguments
        ---------
        Refer to sklearn.ensemble.RandomForestClassifier.predic_proba
        """
        output = np.zeros((len(X), self.num_categories))

        for tree_index in np.arange(self.num_trees):
            proba = np.zeros((len(X), self.num_categories))
            children_left = self.trees[tree_index]["children_left"]
            children_right = self.trees[tree_index]["children_right"]
            features = self.trees[tree_index]["feature"]
            thresholds = self.trees[tree_index]["threshold"]
            tree_proba = self.trees[tree_index]["proba"]
            indexs = np.arange(X.shape[0], dtype=int)
            if len(children_left) > sys.getrecursionlimit():
                sys.setrecursionlimit(int(len(children_left) * 1.2))
            search_nodes(X, children_left, children_right, features, thresholds,
                         tree_proba, proba, indexs, 0)
            output += proba

        output /= self.num_trees

        return output

    def to_fits_hdu(self, index):
        """ Formats tree as a fits Header Data Unit

        Arguments
        ---------
        index : int
        Index of the tree to format

        Return
        ------
        names: list of str
        Names of the different variables

        cols: list of arrays
        Data of the different variables
        """
        # create HDU columns
        names = [
            "children_left", "children_right", "feature", "threshold", "proba"
        ]
        cols = [self.trees[index].get(name) for name in names]

        # create HDU and return
        return names, cols


if __name__ == '__main__':
    pass
