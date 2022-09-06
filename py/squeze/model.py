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
import fitsio

from squeze.defaults import RANDOM_STATE
from squeze.defaults import RANDOM_FOREST_OPTIONS
from squeze.random_forest_classifier import RandomForestClassifier
from squeze.utils import save_json, deserialize


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
        if np.isnan(row["PROB_CLASS3"]):
            prob = np.nan
        else:
            prob = row["PROB_CLASS3"] + row["PROB_CLASS30"]
    elif "PROB_CLASS30" in columns:
        if np.isnan(row["PROB_CLASS30"]):
            prob = np.nan
        else:
            prob = row["PROB_CLASS30"]
    elif "PROB_CLASS3" in columns:
        if np.isnan(row["PROB_CLASS3"]):
            prob = np.nan
        else:
            prob = row["PROB_CLASS3"]
    else:
        prob = np.nan
    return prob


class Model(object):
    """ Create, train and/or execute the quasar model to find quasars

        CLASS: Model
        PURPOSE: Create, train and/or execute the quasar model to find
        quasars
        """

    def __init__(self,
                 name,
                 selected_cols,
                 settings,
                 model_options=(RANDOM_FOREST_OPTIONS, RANDOM_STATE)):
        """ Initialize class instance.

            Parameters
            ----------
            name : string
            Name of the model

            selected_cols : list
            List of the columns to be considered for training

            settings : dict
            A dictionary containing the settings used to find the
            candidates

            random_state : int - Default: RANDOM_STATE
            Integer to set the sandom states of the classifier

            model_options : (dict, int, :) - Defaut: (RANDOM_FOREST_OPTIONS, RANDOM_STATE)
            The first dictionary sets the options to be passed to the random forest
            cosntructor. If high-low split of the training is desired, the
            dictionary must contain the entries "high" and "low", and the
            corresponding values must be dictionaries with the options for each
            of the classifiers. The second int is the random state passed to the
            random forest classifiers. If the tuple has more elements, they are
            ignored.
            """
        self.__name = name
        self.__settings = settings
        self.__selected_cols = selected_cols
        self.__clf_options = model_options[0]
        self.__random_state = model_options[1]
        if "high" in self.__clf_options.keys(
        ) and "low" in self.__clf_options.keys():
            self.__highlow_split = True
        else:
            self.__clf_options = {"all": model_options[0]}
            self.__highlow_split = False

        # load models
        if self.__highlow_split:
            self.__clf_options.get("high")["random_state"] = self.__random_state
            self.__clf_options.get("low")["random_state"] = self.__random_state
            self.__clf_high = RandomForestClassifier(
                **self.__clf_options.get("high"))
            self.__clf_low = RandomForestClassifier(
                **self.__clf_options.get("low"))
        else:
            self.__clf_options.get("all")["random_state"] = self.__random_state
            self.__clf = RandomForestClassifier(**self.__clf_options.get("all"))

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
            if self.__highlow_split:
                class_labels = self.__clf_high.classes_
            else:
                class_labels = self.__clf.classes_
            for class_label in class_labels:
                if row[f"PROB_CLASS{int(class_label):d}"] > aux_prob:
                    aux_prob = row[f"PROB_CLASS{int(class_label):d}"]
                    data_class = int(class_label)

        return data_class

    def get_settings(self):
        """ Access function for self.__settings """
        return self.__settings

    def save_model(self):
        """ Save the model"""

        if self.__name.endswith(".json"):
            save_json(self.__name, self)
        else:
            self.save_model_as_fits()

    def save_model_as_fits(self):
        """ Save the model as a fits file"""
        results = fitsio.FITS(self.__name.replace(".json", ".fits.gz"),
                              'rw',
                              clobber=True)

        # Create settings HDU to store items in self.__settings
        header = [
            {
                "name": "Z_PREC",
                "value": self.__settings.get("Z_PRECISION"),
                "comment": "z_try correct if in z_true +/- Z_PRECISION",
            },
            {
                "name": "PF_WIDTH",
                "value": self.__settings.get("PEAKFIND_WIDTH"),
                "comment": "smoothing used by the peak finder",
            },
            {
                "name": "PF_SIG",
                "value": self.__settings.get("PEAKFIND_SIG"),
                "comment": "min significance used by the peak finder",
            },
        ]
        # now create the columns to store lines and try_lines.
        lines = self.__settings.get("LINES")
        try_lines = self.__settings.get("TRY_LINES")
        names = ["LINE_NAME"]
        cols = [np.array(lines.index, dtype=str)]

        names += [f"LINE_{col}" for col in lines.columns]
        cols += [lines[col] for col in lines.columns]

        # try lines is stored as an array of booleans
        # (True if the value in LINES_NAME is in try_lines, and
        # false otherwise)
        names += ["TRY_LINES"]
        cols += [lines.index.isin(try_lines)]

        results.write(cols, names=names, header=header, extname="SETTINGS")
        del header, names, cols

        # selected_cols is stored as an array of strings
        names = ["SELECTED_COLS"]
        cols = [np.array(self.__selected_cols, dtype=str)]
        results.write(cols, names=names, extname="RF_COLS")
        del names, cols

        # Create model HDU(s) to store the classifiers
        if self.__highlow_split:
            classifier_names = ["high", "low"]
            classifiers = [self.__clf_high, self.__clf_low]
        else:
            classifier_names = ["all"]
            classifiers = [self.__clf]

        for classifier_name, classifier in zip(classifier_names, classifiers):
            header = [{
                "name": key,
                "value": value,
            } for key, value in self.__clf_options.get(classifier_name).items()]
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

            num_trees = classifier.num_trees()
            header += [{
                "name": "N_TREES",
                "value": num_trees,
            }, {
                "name": "N_CAT",
                "value": classifier.num_categories(),
            }]

            names = ["CLASSES"]
            cols = [classifier.classes_]

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
            data_frame_high = data_frame[data_frame["Z_TRY"] >= 2.1].copy()
            if data_frame_high.shape[0] > 0:
                aux = data_frame_high.fillna(-9999.99)
                data_vector = aux[self.__selected_cols[:-2]].values
                data_class_probs = self.__clf_high.predict_proba(data_vector)

                # save the probability for each of the classes
                for index, class_label in enumerate(self.__clf_high.classes_):
                    data_frame_high[
                        f"PROB_CLASS{int(class_label):d}"] = data_class_probs[:,
                                                                              index]

            # low-z split
            # compute probabilities for each of the classes
            data_frame_low = data_frame[(data_frame["Z_TRY"] < 2.1)].copy()
            if data_frame_low.shape[0] > 0:
                aux = data_frame_low.fillna(-9999.99)
                data_vector = aux[self.__selected_cols[:-2]].values
                data_class_probs = self.__clf_low.predict_proba(data_vector)

                # save the probability for each of the classes
                for index, class_label in enumerate(self.__clf_low.classes_):
                    data_frame_low[
                        f"PROB_CLASS{int(class_label):d}"] = data_class_probs[:,
                                                                              index]

            # non-peaks
            data_frame_nonpeaks = data_frame[data_frame["Z_TRY"].isna()].copy()
            if data_frame_nonpeaks.shape[0] > 0:
                # save the probability for each of the classes
                for index, class_label in enumerate(self.__clf_low.classes_):
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
                data_vector = data_frame_peaks[
                    self.__selected_cols[:-2]].fillna(-9999.99).values
                data_class_probs = self.__clf.predict_proba(data_vector)

                # save the probability for each of the classes
                for index, class_label in enumerate(self.__clf.classes_):
                    data_frame_peaks[
                        f"PROB_CLASS{int(class_label):d}"] = data_class_probs[:,
                                                                              index]

            # non-peaks
            data_frame_nonpeaks = data_frame[data_frame["Z_TRY"].isna()].copy()
            if not data_frame_nonpeaks.shape[0] == 0:
                # save the probability for each of the classes
                for index, class_label in enumerate(self.__clf.classes_):
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
        if self.__highlow_split:
            # high-z split
            data_frame_high = data_frame[data_frame["Z_TRY"] >= 2.1].fillna(
                -9999.99)
            data_vector = data_frame_high[self.__selected_cols[:-2]].values
            data_class = data_frame_high.apply(self.__find_class,
                                               axis=1,
                                               args=(True,))
            self.__clf_high.fit(data_vector, data_class)
            # low-z split
            data_frame_low = data_frame[(data_frame["Z_TRY"] < 2.1) & (
                data_frame["Z_TRY"] >= 0.0)].fillna(-9999.99)
            data_vector = data_frame_low[self.__selected_cols[:-2]].values
            data_class = data_frame_low.apply(self.__find_class,
                                              axis=1,
                                              args=(True,))
            self.__clf_low.fit(data_vector, data_class)

        else:
            data_frame = data_frame[(data_frame["Z_TRY"] >= 0.0
                                    )][self.__selected_cols].fillna(-9999.99)
            data_vector = data_frame[self.__selected_cols[:-2]].values
            data_class = data_frame.apply(self.__find_class,
                                          axis=1,
                                          args=(True,))
            self.__clf.fit(data_vector, data_class)

    @classmethod
    def from_json(cls, data):
        """ This function deserializes a json string to correclty build the class.
            It uses the deserialization function of class SimpleSpectrum to reconstruct
            the instances of Spectrum. For this function to work, data should have been
            serialized using the serialization method specified in `save_json` function
            present on `utils.py` """

        # create instance using the constructor
        name = data.get("_Model__name")
        selected_cols = data.get("_Model__selected_cols")
        selected_cols = [col.upper() for col in selected_cols]
        settings = {
            key.upper(): value
            for key, value in data.get("_Model__settings").items()
        }
        lines = deserialize(settings.get("LINES"))
        lines.columns = [col.upper() for col in lines.columns]
        settings["LINES"] = lines
        model_options = [
            data.get("_Model__clf_options"),
            data.get("_Model__random_state")
        ]
        cls_instance = cls(name,
                           selected_cols,
                           settings,
                           model_options=model_options)

        # now update the instance to the current values
        if "high" in model_options[0].keys() and "low" in model_options[0].keys(
        ):
            cls_instance.set_clf_high(
                RandomForestClassifier.from_json(data.get("_Model__clf_high")))
            cls_instance.set_clf_low(
                RandomForestClassifier.from_json(data.get("_Model__clf_low")))
        else:
            cls_instance.set_clf(
                RandomForestClassifier.from_json(data.get("_Model__clf")))

        return cls_instance

    @classmethod
    def from_fits(cls, filename):
        """ This function loads the model information from a fits file.
            The expected shape for the fits file is that provided by the
            function save_model_as_fits.

            Parameters
            ----------
            filename : string
            Name of the fits file

            """
        hdul = fitsio.FITS(filename)

        name = filename.replace("fits.gz", "json")

        # load lines
        selected_cols = [
            sel_col.strip() for sel_col in hdul["RF_COLS"]["SELECTED_COLS"][:]
        ]

        cols = {
            "LINE_NAME": "LINE",
            "LINE_WAVE": "WAVE",
            "LINE_START": "START",
            "LINE_END": "END",
            "LINE_BLUE_START": "BLUE_START",
            "LINE_BLUE_END": "BLUE_END",
            "LINE_RED_START": "RED_START",
            "LINE_RED_END": "RED_END",
        }
        dtypes = [
            str, np.float64, np.float64, np.float64, np.float64, np.float64,
            np.float64, np.float64
        ]
        dat = {
            col.upper(): hdul["SETTINGS"][col][:].astype(dtype)
            for col, dtype in zip(cols, dtypes)
        }
        lines = pd.DataFrame(dat)
        lines = lines.rename(columns=cols).set_index("LINE")

        # load try_lines
        pos = np.where(hdul["SETTINGS"]["TRY_LINES"][:])
        try_lines = [
            try_line.strip() for try_line in hdul["SETTINGS"]["LINE_NAME"][pos]
        ]

        # load settings used to find the candidates
        header = hdul["SETTINGS"].read_header()
        settings = {
            "LINES": lines,
            "TRY_LINES": try_lines,
            "Z_PRECISION": header["Z_PREC"],
            "PEAKFIND_WIDTH": header["PF_WIDTH"],
            "PEAKFIND_SIG": header["PF_SIG"],
        }

        # now load model options
        # case 1: highlow_split
        try:
            high = {}
            header = hdul["HIGHINFO"].read_header()
            for key in header:
                if key in [
                        "XTENSION", "BITPIX", "NAXIS", "NAXIS1", "NAXIS2",
                        "PCOUNT", "GCOUNT", "TFIELDS", "EXTNAME", "COMMENT",
                        "TTYPE1", "TFORM1", "N_TREES", "N_CAT", "random_state"
                ]:
                    continue
                high[key.lower()] = header[key]
            low = {}
            header = hdul["LOWINFO"].read_header()
            for key in header:
                if key in [
                        "XTENSION", "BITPIX", "NAXIS", "NAXIS1", "NAXIS2",
                        "PCOUNT", "GCOUNT", "TFIELDS", "EXTNAME", "COMMENT",
                        "TTYPE1", "TFORM1", "N_TREES", "N_CAT", "random_state"
                ]:
                    continue
                low[key.lower()] = header[key]
            model_options = [{"high": high, "low": low}, header["random_state"]]
        except OSError:
            all_candidates = {}
            header = hdul["ALLINFO"].read_header()
            for key in header:
                if key in [
                        "XTENSION", "BITPIX", "NAXIS", "NAXIS1", "NAXIS2",
                        "PCOUNT", "GCOUNT", "TFIELDS", "EXTNAME", "COMMENT",
                        "TTYPE1", "TFORM1", "N_TREES", "N_CAT", "random_state"
                ]:
                    continue
                all_candidates[key.lower()] = header[key]
            model_options = [all_candidates, header["random_state"]]

        # create instance using the constructor
        cls_instance = cls(name,
                           selected_cols,
                           settings,
                           model_options=model_options)

        # now update the instance to the current values
        if "high" in model_options[0] and "low" in model_options[0]:
            cls_instance.set_clf_high(
                RandomForestClassifier.from_fits_hdul(
                    hdul, "high", "HIGHINFO",
                    args=model_options[0].get("high")))
            cls_instance.set_clf_low(
                RandomForestClassifier.from_fits_hdul(
                    hdul, "low", "LOWINFO", args=model_options[0].get("low")))
        else:
            cls_instance.set_clf(
                RandomForestClassifier.from_fits_hdul(hdul,
                                                      "all",
                                                      "ALLINFO",
                                                      args=model_options[0]))

        hdul.close()
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
