"""
    SQUEzE
    ======

    This file provides useful functions that are used throughout SQUEzE to
    avoid duplicate code
    """
__author__ = "Ignasi Perez-Rafols (iprafols@gmail.com)"
__version__ = "0.1"

from __future__ import print_function
import pickle

def save_pkl(filename, user_object):
    """ Saves object into filename. Encoding file as a python object """
    save_file = open(filename, 'wb')
    pickle.dump(user_object, save_file)
    save_file.close()

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
