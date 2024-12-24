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
from squeze.config import Config


def main():
    """ This function pretrains SQUEzE using BOSS data and creates the json file
        associated with the trained model"""

    config = Config(
        config_dict={
            "general": {
                "output": "../data/BOSS_train_64plates.fits.gz"
            },
            "candidates": {
                "load candidates": True,
                "input candidates": "../data/BOSS_train_64plates.fits.gz",
            }
        })
    candidates = Candidates(config)
    candidates.train_model()

    verboseprint("Done")


if __name__ == '__main__':
    main()
