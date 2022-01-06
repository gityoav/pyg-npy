import os
import shutil
import re
import subprocess
import csv

__all__ = ['path_name', 'path_dirname', 'path_join', 'mkdir']


def path_name(path):
    """
    converts path name to //server/file.extension
    replaces'\' with '/' and ensures no double // other than at start

    :Parameters:
    ----------------
    path : str
        file name.

    :Returns:
    -------
    path : str
        standard name.

    """
    path = path.replace('\\', '/')
    while '//' in path[2:]:
        path = path[:2] + path[2:].replace('//', '/')
    return path
    
def path_dirname(path):
    return path_name(os.path.dirname(path))
    
def path_join(*args):
    return path_name(os.path.join(*args))

def mkdir(path):
    """
    makes a new directory if not exists. It works if path is a filename too.
    """

    directory = path_dirname(path)
    if not os.path.exists(directory):
        try:
            os.makedirs(directory)
        except FileExistsError:
            pass
    return directory
