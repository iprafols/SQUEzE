"""
    SQUEzE
    ======

    This file implements the class Model, that is used to store, train, and
    execute the quasar finding model
"""
__author__ = "Ignasi Perez-Rafols (iprafols@gmail.com)"
__version__ = "0.1"

from sklearn import svm
from sklearn import preprocessing

from squeze_common_functions import save_pkl
from squeze_defaults import SVMS
from squeze_defaults import RANDOM_STATES

class Model(object):
    """ Create, train and/or execute the quasar model to find quasars

        CLASS: Model
        PURPOSE: Create, train and/or execute the quasar model to find
        quasars
        """

    def __init__(self, name, settings, svms=(SVMS, RANDOM_STATES)):
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

            svms : (dict, dict) - Defaut: (SVMS, RANDOM_STATES)
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
        svc_list = []
        for index, selected_cols in self.__svms.items():
            data_frame_aux = data_frame[selected_cols].dropna()
            data_vector = data_frame_aux[selected_cols[:-2]].values
            data_vector = self.__scalers.get(index).transform(data_vector)
            data_frame["SVC{}".format(index)] = -1
            data_class_prob = self.__clfs.get(index).predict_proba(data_vector)
            if self.__clfs.get(index).classes_.size == 3:
                data_frame.at[data_frame_aux.index, "SVC{}".format(index)] = data_class_prob[:, 2]
            svc_list.append("SVC{}".format(index))

        data_frame["prob"] = data_frame[svc_list].max(axis=1)
        data_frame["duplicated"] = data_frame.sort_values([
            "specid", "prob"], ascending=False).duplicated(subset=("specid", "z_true"),
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
        for index, selected_cols in self.__svms.items():
            def find_class(row):
                """ Returns 2 if row is correct, 1 if row is a quasar line, and 0 otherwise"""
                if row["is_correct"]:
                    data_class = 2
                elif row["is_line"]:
                    data_class = 1
                else:
                    data_class = 0
                return data_class
            data_frame_aux = data_frame[selected_cols].dropna()
            data_vector = data_frame_aux[selected_cols[:-2]].values
            scaler = preprocessing.StandardScaler().fit(data_vector)
            data_vector = scaler.transform(data_vector)
            data_class = data_frame_aux.apply(find_class, axis=1)
            clf = svm.SVC(probability=True, class_weight="balanced",
                          random_state=self.__random_states.get(index))
            clf.fit(data_vector, data_class)

            self.__clfs[index] = clf
            self.__scalers[index] = scaler

if __name__ == '__main__':
    pass
