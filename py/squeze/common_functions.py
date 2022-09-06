"""
    SQUEzE
    ======

    This file provides useful functions that are used throughout SQUEzE to
    avoid duplicate code
    """
__author__ = "Ignasi Perez-Rafols (iprafols@gmail.com)"
__version__ = "0.1"

import importlib

import json
import pandas as pd
import numpy as np

def class_from_string(class_name, module_name):
    """Return a class from a string. The class must be saved in a module
    under squeze with the same name as the class but
    lowercase and with and underscore. For example class 'MyClass' should
    be in module squeze.my_class

    Arguments
    ---------
    class_name: str
    Name of the class to load

    module_name: str
    Name of the module containing the class

    Return
    ------
    class_object: Class
    The loaded class

    deafult_args: dict
    A dictionary with the default options (empty for no default options)

    accepted_options: list
    A list with the names of the accepted options

    Raise
    -----
    ImportError if module cannot be loaded
    AttributeError if class cannot be found
    """
    # load module
    module_object = importlib.import_module(module_name)
    # get the class
    class_object = getattr(module_object, class_name)
    # get the dictionary with the default arguments
    try:
        default_args = getattr(module_object, "defaults")
    except AttributeError:
        default_args = {}
    # get the list with the valid options
    try:
        accepted_options = getattr(module_object, "accepted_options")
    except AttributeError:
        accepted_options = []
    return class_object, default_args, accepted_options

def function_from_string(function_name, module_name):
    """Return a function from a string. The class must be saved in a module
    under squeze. For example squeze.utils

    Arguments
    ---------
    function_name: str
    Name of the function to load

    module_name: str
    Full name of the module containing the class. E.g. squeze.utils

    Return
    ------
    function_object: function
    The loaded function

    Raise
    -----
    ImportError if module cannot be loaded
    AttributeError if class cannot be found
    """
    # load module
    module_object = importlib.import_module(module_name)
    # get the class
    function_name = getattr(module_object, function_name)

    return function_name

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
        encodable_object: Object
        An encodable version of the object

        Raises
        ------
        TypeError upon unsuccesful serialization
        """
    encodable_object = None
    # first deal with  all special types of numpy arrays (mandatory since they
    # also inherit from np.ndarray)
    if isinstance(obj, np.ma.core.MaskedArray):
        encodable_object = {
            "np.ma.core.MakedArray": {
                "data": obj.data.tolist(),
                "mask": obj.mask.tolist(),
                "dtype": obj.dtype
            }
        }
    # now deal with normal numpy arrays
    elif isinstance(obj, np.ndarray):
        encodable_object = {
            "np.ndarray": {
                "data": obj.tolist(),
                "dtype": obj.dtype
            }
        }

    # deal with numpy ints
    elif isinstance(obj, np.int64):
        encodable_object = int(obj)
    elif isinstance(obj, np.int32):
        encodable_object = int(obj)

    # deal with numpy floats
    elif isinstance(obj, np.float32):
        encodable_object = float(obj)

    # deal with numpy bools
    elif isinstance(obj, np.bool_):
        encodable_object = bool(obj)

    # deal with other numpy objects
    elif isinstance(obj, np.dtype):
        encodable_object = str(obj)

    # deal with pandas objects
    elif isinstance(obj, pd.DataFrame):
        encodable_object = {"pd.DataFrame": obj.to_json()}

    # deal with complex objects
    elif hasattr(obj, "__dict__"):
        encodable_object = obj.__dict__

    # raise error if the object serialization is not addressed by this class
    if encodable_object is None:
        obj_type = str(type(obj))
        if obj_type.startswith("<class '"):
            obj_type = obj_type[8:-2]
        raise TypeError(f"Object of type {obj_type} is not JSON serializable")
    return encodable_object


def deserialize(json_dict):
    """ Deserializes json dictionary. The dictionary must contain only one item
        which has to be either an array or a pandas DataFrame. For serialization
        of more complex objects, prefer the class method from_json of the respective
        object

        Parameters
        ----------
        json_dict : dict
        Object to deserialize

        Returns
        -------
        my_object : Object
        The object
        """
    my_object = None
    if "np.ndarray" in json_dict:
        aux = json_dict.get("np.ndarray")
        my_object = np.array(aux.get("data"), dtype=aux.get("dtype"))
    if "np.ma.core.MakedArray" in json_dict:
        aux = json_dict.get("np.ma.core.MakedArray")
        my_object = np.ma.array(aux.get("data"), mask=aux.get("mask"))
    if "pd.DataFrame" in json_dict:
        my_object = pd.read_json(json_dict.get("pd.DataFrame"))
    return my_object


def save_json(filename, user_object):
    """ Saves object into filename. Encoding file as a json object.
        Complex object are saved using their __dict__ property"""
    with open(filename, 'w', encoding="UTF-8") as outfile:
        json.dump(user_object, outfile, indent=0, default=serialize)


def load_json(filename):
    """ Loads object from filename. File must be encoded as a json object

        Returns
        -------
        The loaded object
        """
    with open(filename, encoding="UTF-8") as json_file:
        user_object = json.load(json_file)
    return user_object


def verboseprint(*args):
    """ Print each argument separately so caller doesn't need to
        stuff everything to be printed into a single string
        """
    for arg in args:
        print(arg, end=" ")
    print("")


def quietprint(*args):  # pylint: disable=unused-argument
    """ Don't print anything
        """


if __name__ == '__main__':
    pass
