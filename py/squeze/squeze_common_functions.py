from __future__ import print_function
"""
    SQUEzE
    ======

    This file provides useful functions that are used throughout SQUEzE to
    avoid duplicate code
    """
__author__ = "Ignasi Perez-Rafols (iprafols@gmail.com)"
__version__ = "0.1"

import pickle
import json
import pandas as pd
import numpy as np

def save_pkl(filename, user_object):
    """ Saves object into filename. Encoding file as a python object """
    save_file = open(filename, 'wb')
    pickle.dump(user_object, save_file)
    save_file.close()

def save_json(filename, user_object):
    """ Saves object into filename. Encoding file as a json object.
        Complex object are saved using their __dict__ property"""
    def default(object):
        if type(object) == np.ndarray:
            object_dict = {}
            object_dict["data"] = object.tolist()
            return object_dict
        elif type(object) == np.ma.core.MaskedArray:
            object_dict = {}
            object_dict["data"] = object.data.tolist()
            object_dict["mask"] = object.mask.tolist()
            return object_dict
        elif hasattr(object, "__dict__"):
            return object.__dict__
        else:
            obj_type = str(type(object))
            if obj_type.startswith("<class '"):
                obj_type = obj_type[8:-2]
            raise TypeError("Object of type {} is not JSON serializable".format(obj_type))
    with open(filename, 'w') as outfile:
        json.dump(user_object, outfile, indent=4,
                  default=lambda o: o.__dict__)

def save_pd(filename, user_object):
    """ Saves pandas data frame into csv."""
    user_object.to_csv(filename, index=False)

def load_pkl(filename):
    """ Loads object from filename. File must be encoded as a python object

        Returns
        -------
        The loaded object.
        """
    load_file = open(filename, 'rb')
    user_object = pickle.load(load_file)
    load_file.close()
    return user_object

def load_json(filename):
    """ Loads object from filename. File must be encoded as a json object
         
        Returns
        -------
        The loaded object
        """
    with open(filename) as json_file:
        user_object = json.load(json_file)
    return user_object

def load_array_from_json(json_dict):
    """ Formats a json deserialized dict into an array. Takes into account
        if the array was a np.ndarray or a np.ma.core.MaskedArray
        
        Returns
        -------
        The formatted object
        """
    if json_dict.get("mask", None) is None:
        array = np.array(json_dict.get("data"))
    else:
        array = np.ma.array(json_dict.get("data"), mask=json_dict.get("mask"))
    return array

def load_pd(filename):
    """ Loads pandas DataFrame from filename. File must be a csv file
        
        Returns
        -------
        The loaded object
        """
    user_object = pd.read_csv(filename)
    return user_object

def verboseprint(*args):
    """ Print each argument separately so caller doesn't need to
        stuff everything to be printed into a single string
        """
    for arg in args:
        print(arg, end=" ")
    print("")

def quietprint(*args):
    """ Don't print anything
        """
    # pylint: disable=unused-argument
    pass

if __name__ == '__main__':
    pass
