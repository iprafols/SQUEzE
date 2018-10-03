"""
    SQUEzE
    ======

    This file implements the class Model, that is used to store, train, and
    execute the quasar finding model
"""
__author__ = "Ignasi Perez-Rafols (iprafols@gmail.com)"
__version__ = "0.1"

import numpy as np

import tqdm

from sklearn import svm
from sklearn import preprocessing

from squeze_common_functions import save_pkl
from squeze_defaults import SVMS
from squeze_defaults import RANDOM_STATES
from squeze_defaults import CUTS
from squeze_defaults import CLASS_PREDICTED

class Model(object):
    """ Create, train and/or execute the quasar model to find quasars

        CLASS: Model
        PURPOSE: Create, train and/or execute the quasar model to find
        quasars
        """

    def __init__(self, name, settings, svms=(SVMS, RANDOM_STATES), cuts=CUTS):
        """ Initialize class instance.

            Parameters
            ----------
            name : string
            Name of the model

            settings : dict
            A dictionary with the settings used to construct the list of
            Candidates. It should be the return of the function get_settings
            of the Candidates object

            svms : dict - Default: SVMS
            A dictionary containing the lines included in each of the SVM
            instances that will be used to determine the probability of the
            candidate being a quasar.

            svms : (dict, dict) - Default: (SVMS, RANDOM_STATES)
            The first dictionary sets the lines that will be included in each of the SVM
            instances that will be used to determine the probability of the
            candidate being a quasar. The second dictionary has to have the same keys as
            the first dictionary and be comprised of integers to set the sandom states of
            the SVMs. In training mode, they're passed to the model instance before
            training. Otherwise it's ignored.
            """
        self.__name = name
        self.__settings = settings
        self.__svms = svms[0]
        self.__random_states = svms[1]
        self.__clfs = {}
        self.__scalers = {}
        self.__cuts = cuts
        self.__percentiles = {}
    
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
            for class_label in self.__clfs.get(1).classes_:
                if row["prob_class{}".format(class_label)] > aux_prob:
                    aux_prob = row["prob_class{}".format(class_label)]
                    data_class = class_label

        return data_class

    def __match_cuts(self, row, selected_cols):
        """ Return True if the selected columns have all higher than the respective
            values stored in self.__percentiles and False otherwise.
            This function should be called using the DataFrame.apply function
            qith axis=1.
            
            Parameters
            ----------
            row : pd.Series
            A row in the DataFrame.
            
            selected_cols : list of string
            Names of the columns to compare. Musts be keys of self.__percentiles
            and be present in the DataFrame columns.
            
            Returns
            -------
            True if the selected columns have all higher values than the respective
            values stored in self.__percentiles and False otherwise
            """
        return all([row[col] >= self.__percentiles[col] for col in selected_cols])

    def get_settings(self):
        """ Access function for self.__settings """
        return self.__settings

    def save_model(self):
        """ Save the model"""
        save_pkl(self.__name, self)

    def compute_probability(self, data_frame):
        """ Compute the probability of a list of candidates to be quasars

            Parameters
            ----------
            data_frame : pd.DataFrame
            The dataframe where the probabilities will be predicted
            """
        def find_class(row):
            """ Finds the class the instance belongs to from the highest
                probability.
                """
            data_class = -1
            aux_prob = 0.0
            for class_label in self.__clfs.get(1).classes_:
                if row["prob_class{}".format(class_label)] > aux_prob:
                    aux_prob = row["prob_class{}".format(class_label)]
                    data_class = class_label
            return data_class

        # compute probabilities for each of the SVM instances
        svc_dict = {}
        for index, selected_cols in tqdm.tqdm(self.__svms.items()):
            data_frame_aux = data_frame[selected_cols].dropna()
            data_vector = data_frame_aux[selected_cols[:-2]].values
            data_vector = self.__scalers.get(index).transform(data_vector)
            data_frame["SVC{}".format(index)] = -1
            data_class_prob = self.__clfs.get(index).predict_proba(data_vector)
            for class_index, class_label in enumerate(self.__clfs.get(index).classes_):
                data_frame.at[data_frame_aux.index, "SVC{}_class{}".format(index, class_label)] = data_class_prob[:, class_index]
                if "class{}".format(class_label) not in svc_dict.keys():
                    svc_dict["class{}".format(class_label)] = []
                svc_dict["class{}".format(class_label)].append("SVC{}_class{}".format(index, class_label))
        
        # compute the probability for each of the classes
        for class_label in self.__clfs.get(1).classes_:
            data_frame["prob_class{}".format(class_label)] = data_frame[svc_dict["class{}".format(class_label)]].max(axis=1)
    
        # predict class and find the probability of the candidate being a quasar
        data_frame["class_predicted"] = data_frame.apply(self.__find_class, axis=1,
                                                         args=(False, ))
        data_frame["prob"] = data_frame[["prob_class3", "prob_class30"]].sum(axis=1)

        # apply hard-core cuts
        data_frame["prob_SVM"] = data_frame["prob"].copy()
        for selected_cols in self.__cuts[1]:
            indexs = data_frame[data_frame.apply(self.__match_cuts,
                                                 axis=1, args=(selected_cols, ))].index
            print indexs
            data_frame.loc[indexs, "prob"] = 1
            data_frame.loc[indexs, "class_predicted"] = CLASS_PREDICTED["quasar"]
        
        # flag duplicated instances
        data_frame["duplicated"] = data_frame.sort_values([
            "specid", "prob", "prob_SVM"], ascending=False).duplicated(subset=("specid", "z_true"),
                                                                    keep="first").sort_index()

        return data_frame

    def train(self, data_frame):
        """ Create and train all the instances of SVMs specified in self.__svms
            to estimate the probability of a candidate being a quasar

            Parameters
            ----------
            data_frame : pd.DataFrame
            The dataframe where the SVMs are trained
            """
        # compute percentiles to apply hard-core cuts
        self.__percentiles = {selected_col: data_frame[~data_frame["is_correct"]][selected_col].quantile(self.__cuts[0]) for selected_col in np.unique(self.__cuts[1])}

        # filter data_frame by excluding objects that meet the hard-core cuts
        for selected_cols in self.__cuts[1]:
            data_frame = data_frame[~data_frame.apply(self.__match_cuts,
                                                      axis=1, args=(selected_cols, ))]

        # train SVMs
        for index, selected_cols in tqdm.tqdm(self.__svms.items()):
            def find_class(row):
                """ Finds the class the instance belongs to from class_person.
                    For quasars and galaxies add a new class if the redshift is
                    wrong"""
                if row["class_person"] == 30 and not row["correct_redshift"]:
                    data_class = 305
                elif row["class_person"] == 3 and not row["correct_redshift"]:
                    data_class = 35
                elif row["class_person"] == 4 and not row["correct_redshift"]:
                    data_class = 45
                else:
                    data_class = row["class_person"]
                return data_class

            data_frame_aux = data_frame[selected_cols].dropna()
            data_vector = data_frame_aux[selected_cols[:-2]].values
            scaler = preprocessing.StandardScaler().fit(data_vector)
            data_vector = scaler.transform(data_vector)
            data_class = data_frame_aux.apply(self.__find_class, axis=1, args=(True,))
            clf = svm.SVC(probability=True, class_weight="balanced",
                          random_state=self.__random_states.get(index))
            clf.fit(data_vector, data_class)

            self.__clfs[index] = clf
            self.__scalers[index] = scaler

if __name__ == '__main__':
    pass



