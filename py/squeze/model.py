"""
    SQUEzE
    ======

    This file implements the class Model, that is used to store, train, and
    execute the quasar finding model
"""
__author__ = "Ignasi Perez-Rafols (iprafols@gmail.com)"

import os

import numpy as np
import pandas as pd
import fitsio

from squeze.error import Error
from squeze.random_forest_classifier import RandomForestClassifier
from squeze.utils import save_json, load_json


def find_prob(row, columns):
    """ Find the probability of a instance being a quasar by
        adding the probabilities of classes 3 and 30. If
        the probability for this classes are not found,
        then return np.nan

        Parameters
        ----------
        row : pd.Series
        A row in the DataFrame.

        colums: list of string
        The column labels of the Series.

        Returns
        -------
        The probability of the object being a quasar.
        This probability is the sum of the probabilities for classes
        3 and 30. If one of them is not available, then the probability
        is taken as the other one. If both are unavailable, then return
        np.nan
        """
    if "PROB_CLASS3" in columns and "PROB_CLASS30" in columns:
        prob = row["PROB_CLASS3"] + row["PROB_CLASS30"]
    elif "PROB_CLASS30" in columns:
        prob = row["PROB_CLASS30"]
    elif "PROB_CLASS3" in columns:
        prob = row["PROB_CLASS3"]
    else:
        prob = np.nan
    return prob


class Model:
    """ Create, train and/or execute the quasar model to find quasars

    CLASS: Model
    PURPOSE: Create, train and/or execute the quasar model to find
    quasars
    """

    def __init__(self, config):
        """ Initialize class instance.

        Arguments
        ---------
        config: Config
        A configuration instance
        """
        self.config = config
        model_config = self.config.get_section("model")

        self.name = model_config.get("filename")
        if self.name is None:
            message = "In section [model], variable 'filename' is required"
            raise Error(message)

        selected_cols = model_config.get("selected cols")
        if selected_cols is None:
            message = "In section [model], variable 'selected cols' is required"
            raise Error(message)
        self.selected_cols = selected_cols.split()

        random_state = model_config.getint("random state")
        if random_state is None:
            message = "In section [model], variable 'random state' is required"
            raise Error(message)

        clf_options = model_config.get("random forest options")
        if selected_cols is None:
            message = (
                "In section [model], variable 'random forest options' is required"
            )
            raise Error(message)
        self.clf_options = load_json(os.path.expandvars(clf_options))

        # initialize random forest classifier(s)
        if "high" in self.clf_options.keys() and "low" in self.clf_options.keys(
        ):
            self.highlow_split = True
            self.clf_options.get("high")["random_state"] = random_state
            self.clf_options.get("low")["random_state"] = random_state
            self.clf_high = RandomForestClassifier(
                **self.clf_options.get("high"))
            self.clf_low = RandomForestClassifier(**self.clf_options.get("low"))
        else:
            self.highlow_split = False
            self.clf_options = {"all": self.clf_options}
            self.clf_options.get("all")["random_state"] = random_state
            self.clf = RandomForestClassifier(**self.clf_options.get("all"))

    def __find_class(self, row, train):
        """ Find the class the instance belongs to.

        If train is set to True, then find the class from class_person.
        For quasars and galaxies add a new class if the redshift is wrong.
        If train is False, then find the class the instance belongs
        to from the highest of the computed probability.

        Arguments
        ---------
        row : pd.Series
        A row in the DataFrame.

        train : bool
        If True, then dinf the class from the truth table,
        otherwise find it from the computed probabilities

        Return
        ------
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
            if row["CLASS_PERSON"] == 30 and not row["CORRECT_REDSHIFT"]:
                data_class = 305
            elif row["CLASS_PERSON"] == 3 and not row["CORRECT_REDSHIFT"]:
                data_class = 35
            elif row["CLASS_PERSON"] == 4 and not row["CORRECT_REDSHIFT"]:
                data_class = 45
            else:
                data_class = row["CLASS_PERSON"]

        # find class from the probabilities
        else:
            data_class = -1
            aux_prob = 0.0
            if self.highlow_split:
                class_labels = self.clf_high.classes
            else:
                class_labels = self.clf.classes
            for class_label in class_labels:
                if row[f"PROB_CLASS{int(class_label):d}"] > aux_prob:
                    aux_prob = row[f"PROB_CLASS{int(class_label):d}"]
                    data_class = int(class_label)

        return data_class

    def save_model(self):
        """ Save the model"""

        if self.name.endswith(".json"):
            self.save_model_as_json()
        else:
            self.save_model_as_fits()
        self.save_model_config()

    def save_model_as_json(self):
        """ Save the model as a json file"""
        config = self.config
        del self.config
        save_json(os.path.expandvars(self.name), self)
        self.config = config

    def save_model_as_fits(self):
        """ Save the model as a fits file"""
        results = fitsio.FITS(self.name.replace(".json", ".fits.gz"),
                              'rw',
                              clobber=True)

        # Create model HDU(s) to store the classifiers
        if self.highlow_split:
            classifier_names = ["high", "low"]
            classifiers = [self.clf_high, self.clf_low]
        else:
            classifier_names = ["all"]
            classifiers = [self.clf]

        for classifier_name, classifier in zip(classifier_names, classifiers):
            header = [{
                "name": key,
                "value": value,
            } for key, value in self.clf_options.get(classifier_name).items()]
            if classifier_names != "all":
                header += [{
                    "name":
                        "COMMENT",
                    "value": ("Options passed to the classifier for"
                              f"{classifier_names} redshift quasars. Redshifts "
                              "are split at 2.1")
                }]
            else:
                header += [{
                    "name":
                        "COMMENT",
                    "value": ("Options passed to the classifier for"
                              "all redshift quasars")
                }]

            num_trees = classifier.num_trees
            header += [{
                "name": "N_TREES",
                "value": num_trees,
            }, {
                "name": "N_CAT",
                "value": classifier.num_categories,
            }]

            names = ["CLASSES"]
            cols = [classifier.classes]

            # create HDU
            results.write(cols,
                          names=names,
                          header=header,
                          extname=f"{classifier_name}INFO")
            del header, names, cols

            # append classifier trees in different HDUs
            for index in range(num_trees):
                names, cols = classifier.to_fits_hdu(index)
                results.write(cols,
                              names=names,
                              extname=f"{classifier_name}{index}")
                del names, cols

        # End of model HDU(s)
        results.close()

    def save_model_config(self):
        """ Save the model configuration"""
        if self.name.endswith(".json"):
            outname = os.path.expandvars(self.name.replace(".json", ".ini"))
        else:
            outname = os.path.expandvars(self.name.replace(".fits.gz", ".ini"))
        with open(outname, 'w', encoding="utf-8") as config_file:
            self.config.write(config_file)

    def compute_probability(self, data_frame):
        """ Compute the probability of a list of candidates to be quasars

            Parameters
            ----------
            data_frame : pd.DataFrame
            The dataframe where the probabilities will be predicted
            """

        if self.highlow_split:
            # high-z split
            # compute probabilities for each of the classes
            data_frame_high = data_frame[data_frame["Z_TRY"] >= 2.1].copy()
            if data_frame_high.shape[0] > 0:
                aux = data_frame_high.fillna(-9999.99)
                data_vector = aux[self.selected_cols[:-2]].values
                data_class_probs = self.clf_high.predict_proba(data_vector)

                # save the probability for each of the classes
                for index, class_label in enumerate(self.clf_high.classes):
                    data_frame_high[
                        f"PROB_CLASS{int(class_label):d}"] = data_class_probs[:,
                                                                              index]

            # low-z split
            # compute probabilities for each of the classes
            data_frame_low = data_frame[(data_frame["Z_TRY"] < 2.1)].copy()
            if data_frame_low.shape[0] > 0:
                aux = data_frame_low.fillna(-9999.99)
                data_vector = aux[self.selected_cols[:-2]].values
                data_class_probs = self.clf_low.predict_proba(data_vector)

                # save the probability for each of the classes
                for index, class_label in enumerate(self.clf_low.classes):
                    data_frame_low[
                        f"PROB_CLASS{int(class_label):d}"] = data_class_probs[:,
                                                                              index]

            # non-peaks
            data_frame_nonpeaks = data_frame[data_frame["Z_TRY"].isna()].copy()
            if data_frame_nonpeaks.shape[0] > 0:
                # save the probability for each of the classes
                for index, class_label in enumerate(self.clf_low.classes):
                    data_frame_nonpeaks[
                        f"PROB_CLASS{int(class_label):d}"] = np.nan

            # join datasets
            if (data_frame_high.shape[0] == 0 and
                    data_frame_low.shape[0] == 0 and
                    data_frame_nonpeaks.shape[0] == 0):
                data_frame = data_frame_high
            else:
                data_frame = pd.concat(
                    [data_frame_high, data_frame_low, data_frame_nonpeaks],
                    sort=False)

        else:
            # peaks
            # compute probabilities for each of the classes
            data_frame_peaks = data_frame[data_frame["Z_TRY"] >= 0.0].copy()
            if data_frame_peaks.shape[0] > 0:
                data_vector = data_frame_peaks[self.selected_cols[:-2]].fillna(
                    -9999.99).astype(float).values
                data_class_probs = self.clf.predict_proba(data_vector)

                # save the probability for each of the classes
                for index, class_label in enumerate(self.clf.classes):
                    data_frame_peaks[
                        f"PROB_CLASS{int(class_label):d}"] = data_class_probs[:,
                                                                              index]

            # non-peaks
            data_frame_nonpeaks = data_frame[data_frame["Z_TRY"].isna()].copy()
            if not data_frame_nonpeaks.shape[0] == 0:
                # save the probability for each of the classes
                for index, class_label in enumerate(self.clf.classes):
                    data_frame_nonpeaks[
                        f"PROB_CLASS{int(class_label):d}"] = np.nan

            # join datasets
            if (data_frame_peaks.shape[0] == 0 and
                    data_frame_nonpeaks.shape[0] == 0):
                data_frame = data_frame_peaks
            else:
                data_frame = pd.concat([data_frame_peaks, data_frame_nonpeaks],
                                       sort=False)

        # predict class and find the probability of the candidate being a quasar
        data_frame["CLASS_PREDICTED"] = data_frame.apply(self.__find_class,
                                                         axis=1,
                                                         args=(False,))
        data_frame["PROB"] = data_frame.apply(find_prob,
                                              axis=1,
                                              args=(data_frame.columns,))

        # flag duplicated instances
        data_frame["DUPLICATED"] = data_frame.sort_values(
            ["SPECID", "PROB"],
            ascending=False).duplicated(subset=("SPECID",),
                                        keep="first").sort_index()

        return data_frame

    def train(self, data_frame):
        """ Train all the instances of the classifiers to estimate the probability
            of a candidate being a quasar

            Parameters
            ----------
            data_frame : pd.DataFrame
            The dataframe with which the model is trained
            """
        # train classifier
        if self.highlow_split:
            # high-z split
            data_frame_high = data_frame[data_frame["Z_TRY"] >= 2.1].fillna(
                -9999.99)
            data_vector = data_frame_high[self.selected_cols[:-2]].values
            data_class = data_frame_high.apply(self.__find_class,
                                               axis=1,
                                               args=(True,))
            self.clf_high.fit(data_vector, data_class)
            # low-z split
            data_frame_low = data_frame[(data_frame["Z_TRY"] < 2.1) & (
                data_frame["Z_TRY"] >= 0.0)].fillna(-9999.99)
            data_vector = data_frame_low[self.selected_cols[:-2]].values
            data_class = data_frame_low.apply(self.__find_class,
                                              axis=1,
                                              args=(True,))
            self.clf_low.fit(data_vector, data_class)

        else:
            data_frame = data_frame[(
                data_frame["Z_TRY"]
                >= 0.0)][self.selected_cols].fillna(-9999.99)
            data_vector = data_frame[self.selected_cols[:-2]].values
            data_class = data_frame.apply(self.__find_class,
                                          axis=1,
                                          args=(True,))
            self.clf.fit(data_vector, data_class)

    @classmethod
    def from_file(cls, config, filename):
        """ Construct model from file

        Arguments
        ---------
        config: Config
        A configuration instance

        filename: str
        The name of the json file containing the model. The corresponding
        configuration file (ending with ini extension) must also exist

        Return
        ------
        cls_instance: Model
        The loaded instance
        """
        if filename.endswith(".json"):
            cls_instance = cls.from_json(config, filename)
        else:
            cls_instance = cls.from_fits(config, filename)
        return cls_instance

    @classmethod
    def from_json(cls, config, filename):
        """ This function deserializes a json string to correclty build the class.

        It uses the deserialization function of class SimpleSpectrum to reconstruct
        the instances of Spectrum. For this function to work, data should have been
        serialized using the serialization method specified in `save_json` function
        present on `utils.py`

        Arguments
        ---------
        config: Config
        A configuration instance

        filename: str
        The name of the json file containing the model. The corresponding
        configuration file (ending with ini extension) must also exist

        Return
        ------
        cls_instance: Model
        The loaded instance
        """
        cls_instance = cls(config)

        # now update the instance to the current values
        data = load_json(filename)
        if cls_instance.highlow_split:
            cls_instance.clf_high = RandomForestClassifier.from_json(
                data.get("clf_high"))
            cls_instance.clf_low = RandomForestClassifier.from_json(
                data.get("clf_low"))
        else:
            cls_instance.clf = RandomForestClassifier.from_json(data.get("clf"))

        return cls_instance

    @classmethod
    def from_fits(cls, config, filename):
        """ This function loads the model information from a fits file.

        The expected shape for the fits file is that provided by the
        function save_model_as_fits.

        Arguments
        ---------
        config: Config
        A configuration instance

        filename: str
        The name of the json file containing the model. The corresponding
        configuration file (ending with ini extension) must also exist

        Return
        ------
        cls_instance: Model
        The loaded instance
        """
        cls_instance = cls(config)

        # now update the instance to the current values
        hdul = fitsio.FITS(os.path.expandvars(filename))
        if cls_instance.highlow_split:
            cls_instance.clf_high = RandomForestClassifier.from_fits_hdul(
                hdul,
                "high",
                "HIGHINFO",
                args=cls_instance.clf_options.get("high"))
            cls_instance.clf_low = RandomForestClassifier.from_fits_hdul(
                hdul,
                "low",
                "LOWINFO",
                args=cls_instance.clf_options.get("low"))
        else:
            cls_instance.clf = RandomForestClassifier.from_fits_hdul(
                hdul, "all", "ALLINFO", args=cls_instance.clf_options)

        hdul.close()
        return cls_instance


if __name__ == '__main__':
    pass
