from pyg_npy._file import mkdir
from _collections_abc import dict_keys, dict_values
import pandas as pd
import numpy as np
import os.path
import json
from io import BytesIO, SEEK_END, SEEK_SET
import datetime

_npy = '.npy'
_json = '.json'

__all__ = ['NpyAppendArray', 'pd_to_npy', 'pd_read_npy', 'np_save']


class NpyAppendArray:
    """
    appends/writes numpy arrays to file.

    :Example:
    ----------------
    >>> fname = 'c:/temp/temp.npy'
    >>> arr = np.random.normal(0,1, (100,10))
    >>> with NpyAppendArray(fname) as npa:
    >>>     npa.write(arr)
    >>>     npa.append(arr)    
    >>> assert np.load(fname).shape == (200, 10)
    """
    def __init__(self, filename):
        self.filename = filename
        self.fp = None
        self.__is_init = None

    def make_file_appendable(self):
        """
        if format of file is not amenable to append, resave it 
        """
        if os.path.isfile(self.filename):
            with open(self.filename, mode="rb+") as fp:
                magic = np.lib.format.read_magic(fp)
            if magic != (2, 0):
                arr = np.load(self.filename)
                self.write(arr)

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
        if not os.path.isfile(self.filename):
            self.__is_init = False
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
        """
        writes an array to self.filename, overwriting existing file if there
        """
        fp = self.fp  = open(self.filename, mode="wb")
        self.shape, self.fortran_order, self.dtype = list(arr.shape), False, arr.dtype
        fp.write(self.__create_header_bytes())
        self.header_length = fp.tell()
        arr.tofile(fp)        

    def append(self, arr):
        if not arr.flags.c_contiguous:
            raise NotImplementedError("ndarray needs to be c_contiguous")

        if self.__is_init is None:
            self.__init()
        
        if self.__is_init is False:
            self.write(arr)

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
    
    if not value.flags.c_contiguous:
        value = np.ascontiguousarray(value)
    
    mode = mode[0].lower()
    with NpyAppendArray(path) as f:
        if mode == 'a' and os.path.exists(path):
            f.make_file_appendable()
            f.append(value)
        elif mode in 'wa':
            f.write(value)
        else:
            raise ValueError('mode "%s" needs to be "a" or "w"'%mode)
    return path

def pd_to_npy(value, path, mode = 'w', check = True):
    """
    writes a pandas DataFrame/series to a collection of numpy files as columns.
    Support append rather than overwrite
    
    :Params:
    --------
    value: pd.DataFrame/Series
        value to be saved
    
    path: str
        location of the form c:/test/file.npy

    mode: str 'a/w'
        if 'a', will append to existing files rather than overwrite
        if 'w', will overwrite

    check: bool
        if True and we are appending to an existing file, will check that:
            - column names match
            - index name match
            - will only save data with INDEX GREATER than last index value of existing file

    :Returns:
    ---------
    dict of path, columns and index
    These are the inputs needed by pd_read_npy
    
    :Example:
    ----------
    >>> import numpy as np   
    >>> import pandas as pd
    >>> from pyg_base import drange, eq
    >>> from pyg_npy import *
    >>> path = 'c:/temp/test.npy'
    >>> value = pd.DataFrame(np.random.normal(0,1,(100,10)), drange(-99), list('abcdefghij'))
    >>> res = pd_to_npy(value, path)

    >>> res
    >>> {'path': 'c:/temp/test.npy',
    >>>  'columns': ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j'],
    >>>  'index': None,
    >>>  'latest': 1641427200.0} ## This is in UNIX posix time
    
    >>> df = pd_read_npy('c:/temp/test.npy')    
    >>> assert eq(df, value)
    
    :Underneath the hood:
    --------------------
    The numpy df.values array is here:
    
    >>> assert np.load('c:/temp/test/data.npy').shape == value.shape
    
    The index array is here:
    
    >>> np.load('c:/temp/test/index.npy').shape == (len(value),)

    The metadata is here:
    
    >>> with open('c:/temp/test/metadata.json') as f:
    >>>    j = json.load(f)
    >>> j
    >>> {'path': 'c:/temp/test.npy',
    >>>      'columns': ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j'],
    >>>      'index': None,
    >>>      'latest': 1641427200.0}
    

     :Example: appending SAME data with/out check
    ------------------------------
    >>> res = pd_to_npy(value, path, 'a') # this will do nothing as new data is PRE res['latest']
    >>> df = pd_read_npy('c:/temp/test.npy')    
    >>> assert eq(df, value)
    
    ## now we drop check...

    >>> res = pd_to_npy(value, path, mode = 'a', check = False) 
    >>> df = pd_read_npy('c:/temp/test.npy')    
    >>> assert eq(df, pd.concat([value, value]))
 
    """
    res = dict(path = path)
    if isinstance(value, pd.Series):
        df = pd.DataFrame(value)
        columns = list(df.columns)
        res['columns'] = columns[0]
    else:
        df = value
        res['columns'] = columns = list(df.columns)

    res['index'] = df.index.name
    if path.endswith(_npy):
        path = path[:-len(_npy)]    
    
    jname = path +'/%s%s'%('metadata', _json)  
    mode = mode[0].lower()
    if check and mode == 'a' and os.path.isfile(jname):
        with open(jname, 'r') as fp:
            j = json.load(fp)
        assert j['columns'] == res['columns'], 'column names mismatch %s stored vs %s' %(j['columns'], res['columns'])
        latest = j.get('latest')
        if latest:
            if isinstance(df.index, pd.DatetimeIndex):
                latest = datetime.datetime.utcfromtimestamp(latest)
            df = df[df.index > latest]
    else:
        j = res

    if len(df):
        latest = df.index[-1]
        if isinstance(latest, (np.int64, np.int32, np.int16)):
            latest = int(latest)
        elif isinstance(df.index, pd.DatetimeIndex):
            latest = float(np.datetime64(latest).astype('uint64') / 1e6) ## utc
        j['latest'] = latest
    dname = path +'/%s%s'%('data', _npy)
    iname = path +'/%s%s'%('index', _npy)
    np_save(dname, df.values, mode)
    np_save(iname, df.index.values, mode)
    with open(jname, 'w') as fp:
        json.dump(j, fp)
    return j

pd_to_npy.output = ['path', 'columns', 'index', 'latest']


def np_metadata(path):
    """
    returns the file metadata

    Parameters
    ----------
    path : str
        location of npy file.

    Returns
    -------
    dict
        metadata with 'columns', 'latest', 'path' and 'index'

    """
    if path.endswith(_npy):
        path = path[:-len(_npy)]    
    jname = path +'/%s%s'%('metadata', _json)  
    with open(jname, 'r') as fp:
        j = json.load(fp)
    latest = j.get('latest')
    if latest:
        j['latest'] = datetime.datetime.utcfromtimestamp(latest)
    return j

np_metadata.output = ['path', 'columns', 'index', 'latest']


def np_latest(path):
    """
    returns the latest date in the npy file without opening file
    """
    return np_metadata(path).get('latest')


def pd_read_npy(path, columns = None, index = None, latest = None, **kwargs):
    """
    reads a pandas dataframe/series from a path directory containing npy files

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
    data = np.load(path +'/%s%s'%('data', _npy))
    index_data = np.load(path +'/%s%s'%('index', _npy))
    jname = path +'/%s%s'%('metadata', _json)
    if os.path.isfile(jname):
        with open(jname, 'r') as fp:
            j = json.load(fp)
        columns = columns or j['columns']
        index = index or j['index']    
    res = pd.DataFrame(data, index_data)
    res.index.name = index
    if not isinstance(columns, (list, tuple)):
        res = res[0]
    else:
        res.columns = columns
    return res
