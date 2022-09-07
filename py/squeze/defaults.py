"""
    SQUEzE
    ======

    This file intializes the default values of SQUEzE variables
    """
__author__ = "Ignasi Perez-Rafols (iprafols@gmail.com)"
__version__ = "0.1"


# This variable sets the options to be passed to the random forest classifier
RANDOM_FOREST_OPTIONS = {
    "high": {
        "class_weight": "balanced_subsample",
        "n_jobs": 3,
        "n_estimators": 1000,
        "max_depth": 10,
    },
    "low": {
        "class_weight": "balanced_subsample",
        "n_jobs": 3,
        "n_estimators": 1000,
        "max_depth": 10,
    },
}

# This variable sets the random states of the random forest instances
RANDOM_STATE = 2081487193

# This variable contains the transcription from numerical predicted class to
# named predicted class
CLASS_PREDICTED = {
    "star": 1,
    "quasar": 3,
    "quasar, wrong z": 35,
    "quasar, bal": 30,
    "quasar, bal, wrong z": 305,
    "galaxy": 4,
    "galaxy, wrong z": 45,
}



"""
This variable sets the number of pixels to each side of the peak to be used as
metrics
"""
NUM_PIXELS = 30

if __name__ == '__main__':
    pass
