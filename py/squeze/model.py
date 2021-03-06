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
import astropy.io.fits as fits
from astropy.table import Table

from squeze.common_functions import save_json, deserialize
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

    def __init__(self, name, selected_cols, settings,
                 model_opt=(RANDOM_FOREST_OPTIONS, RANDOM_STATE)):
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

            model_opt : tuple -  Default: (RANDOM_FOREST_OPTIONS, RANDOM_STATE)
            A tuple. First item should be a dictionary with the options to be
            passed to the random forest. If two random forests are to be trained
            for high (>=2.1) and low redshift candidates separately, then the
            dictionary must only contain the keys 'high' and 'low', and the
            corresponding values must be dictionaries with the options for each
            of the classifiers. The second element of the tuple is the random
            state that will be used to initialize the forest.
            """
        self.__name = name
        self.__settings = settings
        self.__selected_cols = selected_cols
        self.__clf_options = model_opt[0]
        self.__random_state = model_opt[1]
        if "high" in self.__clf_options.keys() and "low" in self.__clf_options.keys():
            self.__highlow_split = True
        else:
            self.__clf_options = {"all": model_opt[0]}
            self.__highlow_split = False

        # load models
        if self.__highlow_split:
            self.__clf_options.get("high")["random_state"] = self.__random_state
            self.__clf_options.get("low")["random_state"] = self.__random_state
            self.__clf_high = RandomForestClassifier(**self.__clf_options.get("high"))
            self.__clf_low = RandomForestClassifier(**self.__clf_options.get("low"))
        else:
            self.__clf_options.get("all")["random_state"] = self.__random_state
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
                if row["PROB_CLASS{:d}".format(int(class_label))] > aux_prob:
                    aux_prob = row["PROB_CLASS{:d}".format(int(class_label))]
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

        # Create settings HDU to store items in self.__settings
        header = fits.Header()
        header["Z_PREC"] = self.__settings.get("Z_PRECISION")
        header["PF_WIDTH"] = self.__settings.get("PEAKFIND_WIDTH")
        header["PF_SIG"] = self.__settings.get("PEAKFIND_SIG")
        # now create the columns to store lines and try_lines.
        lines = self.__settings.get("LINES")
        try_lines = self.__settings.get("TRY_LINES")
        cols = [
            fits.Column(name="LINE_NAME",
                        array=lines.index,
                        format="10A",
                        ),
            fits.Column(name="LINE_WAVE",
                        array=lines["WAVE"],
                        format="D",
                        ),
            fits.Column(name="LINE_START",
                        array=lines["START"],
                        format="D",
                        ),
            fits.Column(name="LINE_END",
                        array=lines["END"],
                        format="D",
                        ),
            fits.Column(name="LINE_BLUE_START",
                        array=lines["BLUE_START"],
                        format="D",
                        ),
            fits.Column(name="LINE_BLUE_END",
                        array=lines["BLUE_END"],
                        format="D",
                        ),
            fits.Column(name="LINE_RED_START",
                        array=lines["RED_START"],
                        format="D",
                        ),
            fits.Column(name="LINE_RED_END",
                        array=lines["RED_END"],
                        format="D",
                        ),
            # try lines is stored as an array of booleans
            # (True if the value in LINES_NAME is in try_lines, and
            # false otherwise)
            fits.Column(name="TRY_LINES",
                        array=lines.index.isin(try_lines),
                        format="L",
                        ),

            # selected_cols is stored as an array of strings
            fits.Column(name="SELECTED_COLS",
                        array=self.__selected_cols,
                        format="20A",
                        ),
        ]
        # Create settings HDUs
        hdu = fits.BinTableHDU.from_columns(cols,
                                            name="SETTINGS",
                                            header=header)
        # Update header with more detailed info
        desc = {"Z_PREC": "z_try correct if in z_true +/- Z_PRECISION",
                "PK_WIDTH": "smoothing used by the peak finder",
                "PK_SIG": "min significance used by the peak finder",
                "W_MODE": "deprecated, included for testing",
                "LINE_NAME": "name of the line",
                "LINE_WAVE": "wavelength of the line",
                "LINE_START": "start of the peak interval",
                "LINE_END": "end of the peak interval",
                "LINE_BLUE_START": "start of the blue continuum interval",
                "LINE_BLUE_END": "end of the blue continuum interval",
                "LINE_RED_START": "start of the red continuum interval",
                "LINE_RED_END": "end of the red continuum interval",
                "TRY_LINE": "True if this line is part of try_lines",
                }
        for key in hdu.header:
            hdu.header.comments[key] = desc.get(hdu.header[key], "")
        # End of settings HDU

        # store HDU in HDU list and liberate memory
        hdul = fits.HDUList([fits.PrimaryHDU(), hdu])
        del hdu, header, cols, desc

        # Create model HDU(s) to store the classifiers
        if self.__highlow_split:
            names = ["high", "low"]
            classifiers = [self.__clf_high, self.__clf_low]
        else:
            names = ["all"]
            classifiers = [self.__clf]

        for name, classifier in zip(names, classifiers):
            header = fits.Header()
            for key, value in self.__clf_options.get(name).items():
                header[key] = value
            if name is not "all":
                header["COMMENT"] = ("Options passed to the classifier for"
                                     "{} redshift quasars. Redshifts are"
                                     "split at 2.1"
                                     ).format(name)
            else:
                header["COMMENT"] = ("Options passed to the classifier for"
                                     "all redshift quasars")

            num_trees = classifier.num_trees()
            header["N_TREES"] = num_trees
            header["N_CAT"] = classifier.num_categories()
            cols = [fits.Column(name="CLASSES",
                        array=classifier.classes_,
                        format="D",
                        ),]

            # create HDU
            hdu = fits.BinTableHDU.from_columns(cols,
                                                name="{}INFO".format(name),
                                                header=header)

            # add to HDU list
            hdul.append(hdu)
            # append classifier trees in different HDUs
            [hdul.append(classifier.to_fits_hdu(index,
                                               "{}{}".format(name, index)))
             for index in range(num_trees)]
            del hdu, header
        # End of model HDU(s)

        # save fits file
        hdul.writeto(self.__name.replace(".json", ".fits.gz"), overwrite=True)

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
                    data_frame_high["PROB_CLASS{:d}".format(int(class_label))] = data_class_probs[:,index]

            # low-z split
            # compute probabilities for each of the classes
            data_frame_low = data_frame[(data_frame["Z_TRY"] < 2.1)].copy()
            if data_frame_low.shape[0] > 0:
                aux = data_frame_low.fillna(-9999.99)
                data_vector = aux[self.__selected_cols[:-2]].values
                data_class_probs = self.__clf_low.predict_proba(data_vector)

                # save the probability for each of the classes
                for index, class_label in enumerate(self.__clf_low.classes_):
                    data_frame_low["PROB_CLASS{:d}".format(int(class_label))] = data_class_probs[:,index]

            # non-peaks
            data_frame_nonpeaks = data_frame[data_frame["Z_TRY"].isna()].copy()
            if data_frame_nonpeaks.shape[0] > 0:
                # save the probability for each of the classes
                for index, class_label in enumerate(self.__clf_low.classes_):
                    data_frame_nonpeaks["PROB_CLASS{:d}".format(int(class_label))] = np.nan

            # join datasets
            if (data_frame_high.shape[0] == 0 and data_frame_low.shape[0] == 0 and
                data_frame_nonpeaks.shape[0] == 0):
                data_frame = data_frame_high
            else:
                data_frame = pd.concat([data_frame_high, data_frame_low, data_frame_nonpeaks], sort=False)

        else:
            # peaks
            # compute probabilities for each of the classes
            data_frame_peaks = data_frame[data_frame["Z_TRY"] >= 0.0].copy()
            if data_frame_peaks.shape[0] > 0:
                data_vector = data_frame_peaks[self.__selected_cols[:-2]].fillna(-9999.99).values
                data_class_probs = self.__clf.predict_proba(data_vector)

                # save the probability for each of the classes
                for index, class_label in enumerate(self.__clf.classes_):
                    data_frame_peaks["PROB_CLASS{:d}".format(int(class_label))] = data_class_probs[:,index]

            # non-peaks
            data_frame_nonpeaks = data_frame[data_frame["Z_TRY"].isna()].copy()
            if not data_frame_nonpeaks.shape[0] == 0:
                # save the probability for each of the classes
                for index, class_label in enumerate(self.__clf_low.classes_):
                    data_frame_nonpeaks["PROB_CLASS{:d}".format(int(class_label))] = np.nan

            # join datasets
            if (data_frame_peaks.shape[0] == 0 and data_frame_nonpeaks.shape[0] == 0):
                data_frame = data_frame_peaks
            else:
                data_frame = pd.concat([data_frame_peaks, data_frame_nonpeaks], sort=False)

        # predict class and find the probability of the candidate being a quasar
        data_frame["CLASS_PREDICTED"] = data_frame.apply(self.__find_class, axis=1,
                                                         args=(False, ))
        data_frame["PROB"] = data_frame.apply(self.__find_prob, axis=1,
                                              args=(data_frame.columns, ))

        # flag duplicated instances
        data_frame["DUPLICATED"] = data_frame.sort_values(["SPECID", "PROB"], ascending=False).duplicated(subset=("SPECID",), keep="first").sort_index()

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
            data_frame_high = data_frame[data_frame["Z_TRY"] >= 2.1].fillna(-9999.99)
            data_vector = data_frame_high[self.__selected_cols[:-2]].values
            data_class = data_frame_high.apply(self.__find_class, axis=1, args=(True,))
            self.__clf_high.fit(data_vector, data_class)
            # low-z split
            data_frame_low = data_frame[(data_frame["Z_TRY"] < 2.1) & (data_frame["Z_TRY"] >= 0.0)].fillna(-9999.99)
            data_vector = data_frame_low[self.__selected_cols[:-2]].values
            data_class = data_frame_low.apply(self.__find_class, axis=1, args=(True,))
            self.__clf_low.fit(data_vector, data_class)

        else:
            data_frame = data_frame[(data_frame["Z_TRY"] >= 0.0)][self.__selected_cols].fillna(-9999.99)
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
        selected_cols = [col.upper() for col in selected_cols]
        settings = {key.upper(): value
                    for key, value in data.get("_Model__settings").items()}
        lines = deserialize(settings.get("LINES"))
        lines.columns = [col.upper() for col in lines.columns]
        settings["LINES"] = lines
        model_opt = [data.get("_Model__clf_options"), data.get("_Model__random_state")]
        cls_instance = cls(name, selected_cols, settings,
                           model_opt=model_opt)

        # now update the instance to the current values
        if "high" in model_opt[0].keys() and "low" in model_opt[0].keys():
            cls_instance.set_clf_high(RandomForestClassifier.from_json(data.get("_Model__clf_high")))
            cls_instance.set_clf_low(RandomForestClassifier.from_json(data.get("_Model__clf_low")))
        else:
            cls_instance.set_clf(RandomForestClassifier.from_json(data.get("_Model__clf")))

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
        hdul = fits.open(filename)

        name = filename.replace("fits.gz", "json")

        # load lines
        selected_cols = [sel_col.strip()
                         for sel_col in hdul["SETTINGS"].data["SELECTED_COLS"]]

        cols = {"LINE_NAME": "LINE",
                "LINE_WAVE": "WAVE",
                "LINE_START": "START",
                "LINE_END": "END",
                "LINE_BLUE_START": "BLUE_START",
                "LINE_BLUE_END": "BLUE_END",
                "LINE_RED_START": "RED_START",
                "LINE_RED_END":  "RED_END",}
        dtypes = [str, np.float64, np.float64, np.float64, np.float64,
                  np.float64, np.float64, np.float64]
        pos = np.where(hdul["SETTINGS"].data["LINE_NAME"] != "")
        dat = {col.upper(): hdul["SETTINGS"].data[col][pos].astype(dtype)
               for col, dtype in zip(cols, dtypes)}
        lines = pd.DataFrame(dat)
        lines = lines.rename(columns=cols).set_index("LINE")

        # load try_lines
        pos = np.where((hdul["SETTINGS"].data["TRY_LINES"]) &
                       (hdul["SETTINGS"].data["LINE_NAME"] != ""))
        try_lines = hdul["SETTINGS"].data["LINE_NAME"][pos]
        try_lines = [try_line.strip() for try_line in try_lines]

        # load settings used to find the candidates
        settings = {
            "LINES": lines,
            "TRY_LINES": try_lines,
            "Z_PRECISION": hdul["SETTINGS"].header["Z_PREC"],
            "PEAKFIND_WIDTH": hdul["SETTINGS"].header["PF_WIDTH"],
            "PEAKFIND_SIG": hdul["SETTINGS"].header["PF_SIG"],
        }

        # now load model options
        # cas 1: highlow_split
        try:
            high = {}
            for key in hdul["HIGHINFO"].header:
                if key in ["XTENSION", "BITPIX", "NAXIS", "NAXIS1", "NAXIS2",
                           "PCOUNT", "GCOUNT", "TFIELDS", "EXTNAME", "COMMENT",
                           "TTYPE1", "TFORM1", "N_TREES", "N_CAT",
                           "random_state"]:
                    continue
                high[key.lower()] = hdul["HIGHINFO"].header[key]
            low = {}
            for key in hdul["LOWINFO"].header:
                if key in ["XTENSION", "BITPIX", "NAXIS", "NAXIS1", "NAXIS2",
                           "PCOUNT", "GCOUNT", "TFIELDS", "EXTNAME", "COMMENT",
                           "TTYPE1", "TFORM1", "N_TREES", "N_CAT",
                           "random_state"]:
                    continue
                low[key.lower()] = hdul["LOWINFO"].header[key]
            model_opt = [{"high": high, "low": low},
                         hdul["HIGHINFO"].header["random_state"]]
        except:
            all = {}
            for key in hdul["ALLINFO"].header:
                if key in ["XTENSION", "BITPIX", "NAXIS", "NAXIS1", "NAXIS2",
                           "PCOUNT", "GCOUNT", "TFIELDS", "EXTNAME", "COMMENT",
                           "TTYPE1", "TFORM1", "N_TREES", "N_CAT",
                           "random_state"]:
                    continue
                all[key.lower()] = hdul["HIGHINFO"].header[key]
            model_opt = [all, hdul["ALLINFO"].header["random_state"]]

        # create instance using the constructor
        cls_instance = cls(name, selected_cols, settings,
                           model_opt=model_opt)

        # now update the instance to the current values
        if "high" in model_opt[0].keys() and "low" in model_opt[0].keys():
            cls_instance.set_clf_high(RandomForestClassifier.from_fits_hdul(
                hdul, "high", hdul["HIGHINFO"].header["N_TREES"],
                hdul["HIGHINFO"].header["N_CAT"],
                hdul["HIGHINFO"].data["CLASSES"].astype(np.float64),
                args=model_opt[0].get("high")))
            cls_instance.set_clf_low(RandomForestClassifier.from_fits_hdul(
                hdul, "low", hdul["LOWINFO"].header["N_TREES"],
                hdul["LOWINFO"].header["N_CAT"],
                hdul["LOWINFO"].data["CLASSES"].astype(np.float64),
                args=model_opt[0].get("low")))
        else:
            cls_instance.set_clf(RandomForestClassifier.from_fits_hdul(
                hdul, "all", hdul["ALLINFO"].header["N_TREES"],
                hdul["ALLINFO"].header["N_CAT"],
                hdul["ALLINFO"].data["CLASSES"].astype(np.float64),
                args=model_opt[0]))

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
