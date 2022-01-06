from pyg_npy import np_save, pd_to_npy, pd_read_npy
import numpy as np
import pandas as pd
import pytest
import datetime

dates = [datetime.datetime(2000,1,1) + datetime.timedelta(i) for i in range(100)]

def test_np_save():
    path = 'c:/temp/a.npy'
    value = np.random.normal(0,1,(100,10))
    fname = np_save(path, value)
    res = np.load(fname)    
    assert res.shape == (100,10)
    ## overwrite

    fname = np_save(path, value)
    res = np.load(fname)    
    assert res.shape == (100,10)

    fname = np_save(path, value, 'w')
    res = np.load(fname)    
    assert res.shape == (100,10)
    
    for _ in range(5):
        value = np.random.normal(0,1,(100,10))
        fname = np_save(path, value, 'a')
    res = np.load(fname)    
    assert res.shape == (600,10)


def test_save_works_with_non_contiguous_arrays():
    df = pd.DataFrame([[1,2], [3,4]])    
    arr = df.values
    assert not arr.flags.c_contiguous 
    np_save('c:/temp/test.npy', arr, 'w')
    np_save('c:/temp/test.npy', arr, 'a')
    assert np.load('c:/temp/test.npy').shape == (4,2)

def test_np_save_with_initial_bad_write():
    fname = path = 'c:/temp/a.npy'
    value = np.random.normal(0,1,(100,10))
    np.save(path, value)
    res = np.load(fname)    
    assert res.shape == (100,10)
    
    for _ in range(5):
        value = np.random.normal(0,1,(100,10))
        fname = np_save(path, value, 'a')
    res = np.load(fname)    
    assert res.shape == (600,10)

def test_pd_to_npy_and_back():
    for idx in (range(100), dates):
        value = np.random.normal(0,1,(100,10))
        df = pd.DataFrame(value, columns = list('abcdefghij'), index = idx)
        res = pd_to_npy(value = df, path = 'c:/temp/a.npy', mode = 'w')
        df_ = pd_read_npy(**res)
        np.load('c:/temp/a/data.npy')
        assert np.allclose(df.values, df_.values)

        ## now we append but same values
        value2 = np.random.normal(0,1,(100,10))
        df2 = pd.DataFrame(value2, columns = list('abcdefghij'), index = idx)
        res2 = pd_to_npy(df2, 'c:/temp/a.npy', 'a')
        df2_ = pd_read_npy(**res2)
        assert df2_.shape == (100, 10)
        assert np.allclose(df2_.values, value)    

        ## now we append but no checks
        res3 = pd_to_npy(df2, 'c:/temp/a.npy', 'a', check = False)
        df3_ = pd_read_npy(**res3)
        assert df3_.shape == (200, 10)
        assert np.allclose(df3_.values, np.concatenate([value, value2])) 


