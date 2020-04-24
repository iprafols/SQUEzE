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

from squeze.common_functions import deserialize

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
        trees = [{"children_left": hdu.data["children_left"],
                  "children_right": hdu.data["children_right"],
                  "feature": hdu.data["feature"],
                  "threshold": hdu.data["threshold"],
                  "proba": hdu.data["proba"],
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
                for field, type in [("children_left", "I"),
                                    ("children_right", "I"),
                                    ("feature", "I"),
                                    ("threshold", "I"),
                                    ("proba", "{}E".format(self.__num_categories))]
                ]

        # create HDU and return
        return fits.BinTableHDU.from_columns(cols, name=name)

if __name__ == '__main__':
    pass
