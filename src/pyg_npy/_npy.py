from pyg_npy._file import mkdir
import pandas as pd
import numpy as np

_npy = '.npy'

__all__ = ['pd_to_npy', 'pd_read_npy', 'np_save']


def np_save(path, value, append = False):
    """
    saves an array value in the path specified.
    :Parameters:
    ------------
    path : str
        path to save file
    value: np.ndarray
        value to be saved
    append : bool
        if value is to be appended rather than overwrite        
    """
    mkdir(path)
    if append:
        from npy_append_array import NpyAppendArray as npa
        with npa(path) as f:
            f.append(value)
    else:
        np.save(path, value)
    return path

def pd_to_npy(value, path, append = False):
    """
    writes a pandas DataFrame/series to a collection of numpy files as columns.
    Support append rather than overwrite
    
    :Params:
    --------
    value: pd.DataFrame/Series
        value to be saved
    
    path: str
        location of the form c:/test/file.npy

    append: bool
        if True, will append to existing files rather than overwrite


    :Returns:
    ---------
    dict of path, columns and index
    These are the inputs needed by pd_read_npy
    
    :Example:
    ----------
    >>> import numpy as np   
    >>> import pandas as pd
    >>> from pyg_base import *
    >>> path = 'c:/temp/test.npy'
    >>> value = pd.DataFrame([[1,2],[3,4]], drange(-1), ['a', 'b'])
    >>> res = pd_to_npy(value, path)

    >>> res
    >>> {'path': 'c:/temp/test.npy', 'columns': ['a', 'b'], 'index': ['index']}

    >>> df = pd_read_npy(**res)    
    >>> assert eq(df, value)
    
    """
    res = dict(path = path)
    if isinstance(value, pd.Series):
        df = pd.DataFrame(value)
        columns = list(df.columns)
        res['columns'] = columns[0]
    else:
        df = value
        res['columns'] = columns = list(df.columns)

    df = df.reset_index()
    res['index'] = list(df.columns)[:-len(columns)]    
    if path.endswith(_npy):
        path = path[:-len(_npy)]
    
    for col in df.columns:
        a = df[col].values
        fname = path +'/%s%s'%(col, _npy)
        np_save(fname, a, append)
    return res

pd_to_npy.output = ['path', 'columns', 'index']

def pd_read_npy(path, columns, index):
    """
    reads a pandas dataframe/series from a path directory containing npy files with col.npy and idx.npy names

    Parameters
    ----------
    path : str
        directory where files are.
    columns : str/list of str
        filenames for columns. If columns is a single str, assumes we want a pd.Series
    index : str/list of str
        column names used as indices

    Returns
    -------
    res : pd.DataFrame/pd.Series
    
    """
    if path.endswith(_npy):
        path = path[:-len(_npy)]
    data = {col : np.load(path +'/%s%s'%(col, _npy)) for col in as_list(columns) + as_list(index)}
    res = pd.DataFrame(data).set_index(index)
    if isinstance(columns, str): # it is a series
        res = res[columns]
    return res
