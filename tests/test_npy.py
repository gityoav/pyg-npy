from pyg_npy import np_save, pd_to_npy, pd_read_npy
import numpy as np
import pandas as pd
import pytest


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
    