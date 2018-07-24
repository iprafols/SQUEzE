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

from squeze_common_functions import save_pkl, load_pkl
from squeze_defaults import SVMS
from squeze_defaults import RANDOM_STATES

class Model(object):
    """ Create, train and/or execute the quasar model to find quasars

        CLASS: Model
        PURPOSE: Create, train and/or execute the quasar model to find
        quasars
        """

    def __init__(self, name, settings, svms=SVMS, random_states=RANDOM_STATES):
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
            
            """
        self.__name = name
        self.__settings = settings
        self.__svms = svms
        self.__random_states = random_states

    def get_settings(self):
        """ Access function for self.__settings """
        return self.__settings

    def save_model(self):
        """ Save the model"""
        save_pkl(self.__name, self)

    def compute_probability(self, df):
        """ Compute the probability of a list of candidates to be quasars
            
            Parameters
            ----------
            df : pd.DataFrame
            The dataframe where the probabilities will be predicted
            """
        SVC_list = []
        for index, selected_cols in self.__svms.items():
            df_aux = df[selected_cols].dropna()
            X = df_aux[selected_cols[:-2]].values
            X = self.__scalers.get(index).transform(X)
            df["SVC{}".format(index)] = -1
            y_prob = self.__clfs.get(index).predict_proba(X)
            if self.__clfs.get(index).classes_.size == 3:
                df.at[df_aux.index, "SVC{}".format(index)] = y_prob[:,2]
            SVC_list.append("SVC{}".format(index))

        df["prob"] = df[SVC_list].max(axis=1)
        df["duplicated"] = df.sort_values(["specid", "prob"],
                                          ascending=False).duplicated(subset=("specid", "z_true"),
                                                                      keep="first").sort_index()
        return df

    def train(self, df):
        """ Create and train all the instances of SVMs specified in self.__svms
            to estimate the probability of a candidate being a quasar
            
            Parameters
            ----------
            df : pd.DataFrame
            The dataframe where the SVMs are trained
            """
        self.__clfs = {}
        self.__scalers = {}
        for index, selected_cols in self.__svms.items():
            def find_class(row):
                if row["is_correct"]:
                    return 2
                elif row["is_line"]:
                    return 1
                else:
                    return 0
            df_aux = df[selected_cols].dropna()
            X = df_aux[selected_cols[:-2]].values
            scaler = preprocessing.StandardScaler().fit(X)
            X = scaler.transform(X)
            y = df_aux.apply(find_class, axis=1)
            clf = svm.SVC(probability=True, class_weight="balanced", random_state=self.__random_states.get(index))
            clf.fit(X, y)
    
            self.__clfs[index] = clf
            self.__scalers[index] = scaler

if __name__ == '__main__':
    pass
