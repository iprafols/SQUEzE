"""
    SQUEzE
    ======

    This file implements the class Model, that is used to store, train, and
    execute the quasar finding model
"""
__author__ = "Ignasi Perez-Rafols (iprafols@gmail.com)"
__version__ = "0.1"

import numpy as np
import pandas as pd

from squeze.common_functions import save_json
from squeze.defaults import CUTS
from squeze.defaults import CLASS_PREDICTED
from squeze.defaults import RANDOM_STATE
from squeze.defaults import RANDOM_FOREST_OPTIONS
from squeze.random_forest_classifier import RandomForestClassifier

class Model(object):
    """ Create, train and/or execute the quasar model to find quasars

        CLASS: Model
        PURPOSE: Create, train and/or execute the quasar model to find
        quasars
        """

    def __init__(self, name, selected_cols, settings, cuts=CUTS,
                 model_opt=(RANDOM_FOREST_OPTIONS, RANDOM_STATE)):
        """ Initialize class instance.

            Parameters
            ----------
            name : string
            Name of the model

            settings : dict
            A dictionary with the settings used to construct the list of
            Candidates. It should be the return of the function get_settings
            of the Candidates object

            random_state : int - Default: RANDOM_STATE
            Integer to set the sandom states of the classifier

            clf_options : dict -  Default: RANDOM_FOREST_OPTIONS
            A dictionary with the settings to be passed to the random forest
            constructor. If high-low split of the training is desired, the
            dictionary must contain the entries "high" and "low", and the
            corresponding values must be dictionaries with the options for each
            of the classifiers
            """
        self.__name = name
        self.__settings = settings
        self.__selected_cols = selected_cols
        self.__clf_options = model_opt[0]
        self.__random_state = model_opt[1]
        self.__cuts = cuts
        if "high" in self.__clf_options.keys() and "low" in self.__clf_options.keys():
            self.__highlow_split = True
        else:
            self.__highlow_split = False

        # load models
        if self.__highlow_split:
            self.__clf_options.get("high")["random_state"] = self.__random_state
            self.__clf_options.get("low")["random_state"] = self.__random_state
            self.__clf_high = RandomForestClassifier(**self.__clf_options.get("high"))
            self.__clf_low = RandomForestClassifier(**self.__clf_options.get("low"))
        else:
            self.__clf_options["random_state"] = self.__random_state
            self.__clf = RandomForestClassifier(**self.__clf_options)


    def __find_class(self, row, train):
        """ Find the class the instance belongs to. If train is set
            to True, then find the class from class_person. For quasars
            and galaxies add a new class if the redshift is wrong.
            If train is False, then find the class the instance belongs
            to from the highest of the computed probability.

            Parameters
            ----------
            row : pd.Series
            A row in the DataFrame.

            train : bool
            If True, then dinf the class from the truth table,
            otherwise find it from the computed probabilities

            Returns
            -------
            The class the instance belongs to:
            "star": 1
            "quasar": 3
            "quasar, wrong z": 35
            "quasar, bal": 30
            "quasar, bal, wrong z": 305
            "galaxy": 4
            "galaxy, wrong z": 45
            """
        # find class from the truth table
        if train:
            if row["class_person"] == 30 and not row["correct_redshift"]:
                data_class = 305
            elif row["class_person"] == 3 and not row["correct_redshift"]:
                data_class = 35
            elif row["class_person"] == 4 and not row["correct_redshift"]:
                data_class = 45
            else:
                data_class = row["class_person"]

        # find class from the probabilities
        else:
            data_class = -1
            aux_prob = 0.0
            if self.__highlow_split:
                class_labels = self.__clf_high.classes_
            else:
                class_labels = self.__clf.classes_
            for class_label in class_labels:
                if row["prob_class{:d}".format(int(class_label))] > aux_prob:
                    aux_prob = row["prob_class{:d}".format(int(class_label))]
                    data_class = int(class_label)

        return data_class

    def __find_prob(self, row, columns):
        """ Find the probability of a instance being a quasar by
            adding the probabilities of classes 3 and 30. If
            the probability for this classes are not found,
            then return np.nan

            Parameters
            ----------
            row : pd.Series
            A row in the DataFrame.

            Returns
            -------
            The probability of the object being a quasar.
            This probability is the sum of the probabilities for classes
            3 and 30. If one of them is not available, then the probability
            is taken as the other one. If both are unavailable, then return
            np.nan
            """
        if "prob_class3" in columns and "prob_class30" in columns:
            if row["prob_class3"] == -1.0:
                prob = -1.0
            else:
                prob = row["prob_class3"] + row["prob_class30"]
        elif "prob_class30" in columns:
            if row["prob_class30"] == -1.0:
                prob = -1.0
            else:
                prob = row["prob_class30"]
        elif "prob_class3" in columns:
            if row["prob_class3"] == -1.0:
                prob = -1.0
            else:
                prob = row["prob_class3"]
        else:
            prob = np.nan
        return prob

    def get_settings(self):
        """ Access function for self.__settings """
        return self.__settings

    def save_model(self):
        """ Save the model"""
        save_json(self.__name, self)

    def compute_probability(self, data_frame):
        """ Compute the probability of a list of candidates to be quasars

            Parameters
            ----------
            data_frame : pd.DataFrame
            The dataframe where the probabilities will be predicted
            """

        if self.__highlow_split:
            # high-z split
            # compute probabilities for each of the classes
            data_frame_high = data_frame[data_frame["z_try"] >= 2.1]
            if data_frame_high.shape[0] > 0:
                data_frame_high = data_frame_high.fillna(-9999.99)
                data_vector = data_frame_high[self.__selected_cols[:-2]].values
                data_class_probs = self.__clf_high.predict_proba(data_vector)

                # save the probability for each of the classes
                for index, class_label in enumerate(self.__clf_high.classes_):
                    data_frame_high["prob_class{:d}".format(int(class_label))] = data_class_probs[:,index]

            # low-z split
            # compute probabilities for each of the classes
            data_frame_low = data_frame[(data_frame["z_try"] < 2.1) & (data_frame["z_try"] >= 0.0)]
            if data_frame_low.shape[0] > 0:
                data_frame_low = data_frame_low.fillna(-9999.99)
                data_vector = data_frame_low[self.__selected_cols[:-2]].values
                data_class_probs = self.__clf_low.predict_proba(data_vector)

                # save the probability for each of the classes
                for index, class_label in enumerate(self.__clf_low.classes_):
                    data_frame_low["prob_class{:d}".format(int(class_label))] = data_class_probs[:,index]

            # non-peaks
            data_frame_nonpeaks = data_frame[data_frame["z_try"] == -1.0]
            if data_frame_nonpeaks.shape[0] > 0:
                data_frame_nonpeaks = data_frame_nonpeaks.fillna(-9999.99)
                # save the probability for each of the classes
                for index, class_label in enumerate(self.__clf_low.classes_):
                    data_frame_nonpeaks["prob_class{:d}".format(int(class_label))] = -1.0

            # join datasets
            if (data_frame_high.shape[0] == 0 and data_frame_low.shape[0] == 0 and
                data_frame_nonpeaks.shape[0]) == 0:
                data_frame = data_frame_high
            else:
                data_frame = pd.concat([data_frame_high, data_frame_low, data_frame_nonpeaks])

        else:
            # peaks
            # compute probabilities for each of the classes
            data_frame_peaks = data_frame[data_frame["z_try"] >= 0.0]
            if data_frame_peaks.shape[0] > 0:
                data_vector = data_frame_peaks[self.__selected_cols[:-2]].fillna(-9999.99).values
                data_class_probs = self.__clf.predict_proba(data_vector)

                # save the probability for each of the classes
                for index, class_label in enumerate(self.__clf.classes_):
                    data_frame_peaks["prob_class{:d}".format(int(class_label))] = data_class_probs[:,index]

            # non-peaks
            data_frame_nonpeaks = data_frame[data_frame["z_try"] == -1.0]
            if not data_frame_nonpeaks.shape[0] == 0:
                data_frame_nonpeaks = data_frame_nonpeaks.fillna(-9999.99)
                # save the probability for each of the classes
                for index, class_label in enumerate(self.__clf_low.classes_):
                    data_frame_nonpeaks["prob_class{:d}".format(int(class_label))] = -1.0

            # join datasets
            if data_frame_peaks.shape[0] == 0 and data_frame_nonpeaks.shape[0] == 0:
                data_frame = data_frame_peaks
            else:
                data_frame = pd.concat([data_frame_peaks, data_frame_nonpeaks])

        # predict class and find the probability of the candidate being a quasar
        data_frame["class_predicted"] = data_frame.apply(self.__find_class, axis=1,
                                                         args=(False, ))
        data_frame["prob"] = data_frame.apply(self.__find_prob, axis=1,
                                              args=(data_frame.columns, ))

        # flag duplicated instances
        data_frame["duplicated"] = data_frame.sort_values(["specid", "prob"], ascending=False).duplicated(subset=("specid",), keep="first").sort_index()

        return data_frame

    def train(self, data_frame):
        """ Train all the instances of the classifiers to estimate the probability
            of a candidate being a quasar

            Parameters
            ----------
            data_frame : pd.DataFrame
            The dataframe with which the model is trained
            """
        # filter data_frame by excluding objects that meet the hard-core cuts
        for selected_cols in self.__cuts[1]:
            # train classifier
            if self.__highlow_split:
                # high-z split
                data_frame_high = data_frame[data_frame["z_try"] >= 2.1].fillna(-9999.99)
                data_vector = data_frame_high[self.__selected_cols[:-2]].values
                data_class = data_frame_high.apply(self.__find_class, axis=1, args=(True,))
                self.__clf_high.fit(data_vector, data_class)
                # low-z split
                data_frame_low = data_frame[(data_frame["z_try"] < 2.1) & (data_frame["z_try"] >= 0.0)].fillna(-9999.99)
                data_vector = data_frame_low[self.__selected_cols[:-2]].values
                data_class = data_frame_low.apply(self.__find_class, axis=1, args=(True,))
                self.__clf_low.fit(data_vector, data_class)

            else:
                data_frame = data_frame[(data_frame["z_try"] >= 0.0)][self.__selected_cols].fillna(-9999.99)
                data_vector = data_frame[self.__selected_cols[:-2]].values
                data_class = data_frame.apply(self.__find_class, axis=1, args=(True,))
                self.__clf.fit(data_vector, data_class)

    @classmethod
    def from_json(cls, data):
        """ This function deserializes a json string to correclty build the class.
            It uses the deserialization function of class SimpleSpectrum to reconstruct
            the instances of Spectrum. For this function to work, data should have been
            serialized using the serialization method specified in `save_json` function
            present on `squeze_common_functions.py` """

        # create instance using the constructor
        name = data.get("_Model__name")
        selected_cols = data.get("_Model__selected_cols")
        settings = data.get("_Model__settings")
        cuts = data.get("_Model__cuts")
        model_opt = [data.get("_Model__clf_options"), data.get("_Model__random_state")]
        cls_instance = cls(name, selected_cols, settings, cuts=cuts,
                           model_opt=model_opt)

        # now update the instance to the current values
        if "high" in model_opt[0].keys() and "low" in model_opt[0].keys():
            cls_instance.set_clf_high(RandomForestClassifier.from_json(data.get("_Model__clf_high")))
            cls_instance.set_clf_low(RandomForestClassifier.from_json(data.get("_Model__clf_low")))
        else:
            cls_instance.set_clf(RandomForestClassifier.from_json(data.get("_Model__clf")))

        return cls_instance

    def set_clf_high(self, clf_high):
        """ Set the variable __clf_high. Should only be called from the method from_json"""
        self.__clf_high = clf_high

    def set_clf_low(self, clf_low):
        """ Set the variable __clf_low. Should only be called from the method from_json"""
        self.__clf_low = clf_low

    def set_clf(self, clf):
        """ Set the variable __clf. Should only be called from the method from_json"""
        self.__clf = clf

if __name__ == '__main__':
    pass
