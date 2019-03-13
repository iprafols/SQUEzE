"""
    SQUEzE
    ======
    
    This file allows the user to pretrain SQUEzE using BOSS data
    """
__author__ = "Ignasi Perez-Rafols (iprafols@gmail.com)"
__version__ = "0.1"

import pandas as pd

from squeze_common_functions import save_pkl
from squeze_candidates import Candidates


def main():
    """ This function pretrains SQUEzE using BOSS data and creates the pkl file
        associated with the trained model"""

    print "loading data"
    df = pd.read_csv("../data/BOSS_train_64plates.csv")
    save_pkl("../data/BOSS_train_64plates.pkl", df)

    print "training model"
    candidates = Candidates(mode="training", name="../data/BOSS_train_64plates.pkl")
    candidates.load_candidates("../data/BOSS_train_64plates.pkl")
    candidates.train_model()
    
    print "done"

if __name__ == '__main__':
    main()

