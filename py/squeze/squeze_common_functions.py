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

def serialize(obj):
    """ Serializes complex objects. If the object type is not considered
        for this function, raise a TypeError (as per save_json documentation
        requirements)
        
        Parameters
        ----------
        obj : object
        Object to serialize
        
        Returns
        -------
        Serialized object
        
        Raises
        ------
        TypeError upon unsuccesful serialization
        """
    if isinstance(obj, np.ndarray):
        return {"np.ndarray": {"data": obj.tolist(),
                               "dtype": obj.dtype}}
    if isinstance(obj, np.ma.core.MaskedArray):
        return {"np.ma.core.MakedArray": {"data": obj.data.tolist(),
                                          "maks": obj.mask.tolist(),
                                          "dtype": obj.dtype}}
    if isinstance(obj, np.dtype):
        return str(obj)
    if hasattr(obj, "__dict__"):
        return obj.__dict__

    # raise error if the object serialization is not addressed by this class
    obj_type = str(type(obj))
    if obj_type.startswith("<class '"):
        obj_type = obj_type[8:-2]
    raise TypeError("Object of type {} is not JSON serializable".format(obj_type))
    
def deserialize(json_dict):
    if "np.ndarray" in json_dict:
        obj = json_dict.get("np.ndarray")
        return np.array(obj.get("data"),
                        dtype=obj.get("dtype"))
    if "np.ma.core.MakedArray" in json_dict:
        obj = json_dict.get("np.ma.core.MakedArray")
        return np.ma.array(obj.get("data"), mask=obj.get("mask"))

def save_json(filename, user_object):
    """ Saves object into filename. Encoding file as a json object.
        Complex object are saved using their __dict__ property"""
    with open(filename, 'w') as outfile:
        json.dump(user_object, outfile, indent=4,
                  default=serialize)

def save_pd(filename, user_object):
    """ Saves pandas data frame into csv."""
    user_object.to_csv(filename, index=False)

def load_json(filename):
    """ Loads object from filename. File must be encoded as a json object
         
        Returns
        -------
        The loaded object
        """
    with open(filename) as json_file:
        user_object = json.load(json_file)
    return user_object

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
