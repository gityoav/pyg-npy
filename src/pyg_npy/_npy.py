from pyg_npy._file import mkdir
from _collections_abc import dict_keys, dict_values
import pandas as pd
import numpy as np
import os.path
from io import BytesIO, SEEK_END, SEEK_SET

_npy = '.npy'

__all__ = ['pd_to_npy', 'pd_read_npy', 'np_save']


class NpyAppendArray:
    def __init__(self, filename, mode = 'a'):
        self.filename = filename
        self.fp = None
        self.__is_init = False
        self.mode = mode[0].lower()
        if self.mode not in 'aw':
            raise ValueError('mode can be either append or write')
        if os.path.isfile(filename) and mode == 'a':
            self.__init()

    def __create_header_bytes(self, spare_space = True):
        from struct import pack
        header_map = {
            'descr': np.lib.format.dtype_to_descr(self.dtype),
            'fortran_order': self.fortran_order,
            'shape': tuple(self.shape)
        }
        io = BytesIO()
        np.lib.format.write_array_header_2_0(io, header_map)

        # create array header with 64 byte space space for shape to grow
        io.getbuffer()[8:12] = pack("<I", int(
            io.getbuffer().nbytes-12+(64 if spare_space else 0)
        ))
        if spare_space:
            io.getbuffer()[-1] = 32
            io.write(b" "*64)
            io.getbuffer()[-1] = 10

        return io.getbuffer()

    def __init(self):
        try: 
            return self.__init_from_existing()
        except NotImplementedError:
            arr = np.load(self.filename)
            self.write(arr)
            return self.__init_from_existing()
        
    def __init_from_existing(self):
        if not os.path.isfile(self.filename):
            return

        self.fp = open(self.filename, mode="rb+")
        fp = self.fp
        magic = np.lib.format.read_magic(fp)

        if magic != (2, 0):
            raise NotImplementedError(
                "version (%d, %d) not implemented" % magic
            )

        header = np.lib.format.read_array_header_2_0(fp)
        shape, self.fortran_order, self.dtype = header
        self.shape = list(shape)

        if self.fortran_order == True:
            raise NotImplementedError("fortran_order not implemented")

        self.header_length = fp.tell()

        header_length = self.header_length

        new_header_bytes = self.__create_header_bytes()

        if len(new_header_bytes) != header_length:
            raise TypeError("no spare header space in target file %s" % (
                self.filename
            ))

        self.fp.seek(0, SEEK_END)
        self.__is_init = True

    def __write_header(self):
        fp = self.fp
        fp.seek(0, SEEK_SET)

        new_header_bytes = self.__create_header_bytes()
        header_length = self.header_length

        if header_length != len(new_header_bytes):
            new_header_bytes = self.__create_header_bytes(False)

            # This can only happen if array became so large that header space
            # space is exhausted, which requires more energy than is necessary
            # to boil the earth's oceans:
            # https://hbfs.wordpress.com/2009/02/10/to-boil-the-oceans
            if header_length != len(new_header_bytes):
                raise TypeError(
                    "header length mismatch, old: %d, new: %d" % (
                        header_length, len(new_header_bytes)
                    )
                )

        fp.write(new_header_bytes)
        fp.seek(0, SEEK_END)
        
    def write(self, arr):
        fp = self.fp  = open(self.filename, mode="wb")
        self.shape, self.fortran_order, self.dtype = list(arr.shape), False, arr.dtype
        fp.write(self.__create_header_bytes())
        self.header_length = fp.tell()
        arr.tofile(fp)        

    def append(self, arr):
        if not arr.flags.c_contiguous:
            raise NotImplementedError("ndarray needs to be c_contiguous")

        if not self.__is_init:
            return self.write(arr)

        if arr.dtype != self.dtype:
            raise TypeError("incompatible ndarrays types %s and %s" % (
                arr.dtype, self.dtype
            ))

        shape = self.shape

        if len(arr.shape) != len(shape):
            raise TypeError("incompatible ndarrays shape lengths %s and %s" % (
                len(arr.shape), len(shape)
            ))

        if shape[1:] != list(arr.shape[1:]):
            raise TypeError("ndarray shapes can only differ on zero axis")

        self.shape[0] += arr.shape[0]

        arr.tofile(self.fp)

        self.__write_header()

    def save(self, arr):
        if self.mode == 'a':
            return self.append(arr)
        elif self.mode == 'w':
            return self.write(arr)
        
    def close(self):
        if self.__is_init:
            self.fp.close()

            self.__is_init = False

    def __del__(self):
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.__del__()


def is_rng(value):
    return isinstance(value, (list, tuple, range, dict_keys, dict_values, zip))

def as_list(value, none = False):
    if value is None and not none:
        return []
    elif isinstance(value, list):
        return value
    elif isinstance(value, tuple):
        if len(value)==1 and isinstance(value[0], list):
            return value[0]
        else:
            return list(value)
    elif is_rng(value):
        return list(value)
    else:
        return [value]

def np_save(path, value, mode = 'w'):
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
    with NpyAppendArray(path, mode) as f:
        f.save(value)
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
