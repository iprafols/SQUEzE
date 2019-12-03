"""
    SQUEzE
    ======
    
    This file allows the user to pretrain SQUEzE using BOSS data
    """
__author__ = "Ignasi Perez-Rafols (iprafols@gmail.com)"
__version__ = "0.1"

import pandas as pd

from squeze.squeze_common_functions import verboseprint
from squeze.squeze_candidates import Candidates

def main():
    """ This function pretrains SQUEzE using BOSS data and creates the json file
        associated with the trained model"""

    verboseprint("training model")
    candidates = Candidates(mode="training", name="../data/BOSS_train_64plates.json")
    candidates.load_candidates("../data/BOSS_train_64plates.json")
    candidates.train_model()
    
    verboseprint("done")

if __name__ == '__main__':
    main()

