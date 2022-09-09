#!/usr/bin/python3
"""
    SQUEzE
    ======

    This file allows the user to pretrain SQUEzE using BOSS data
    """
__author__ = "Ignasi Perez-Rafols (iprafols@gmail.com)"
__version__ = "0.1"

import pandas as pd

from squeze.candidates import Candidates
from squeze.utils import verboseprint

def main():
    """ This function pretrains SQUEzE using BOSS data and creates the json file
        associated with the trained model"""

    verboseprint("Loading dataset")
    config = Config()
    config.set_option("general", "output", "../data/BOSS_train_64plates.fits.gz")
    candidates = Candidates(config)
    candidates.load_candidates("../data/BOSS_train_64plates.fits.gz")
    verboseprint("Training model")
    candidates.train_model(False)

    verboseprint("Done")

if __name__ == '__main__':
    main()
