# pyg-npy

pip install from https://pypi.org/project/pyg-npy/

conda install from https://anaconda.org/yoavgit/pyg-npy

A quick utility to save dataframes as npy files. 

It supports append and checks lightly on column names matching and index.

For simple read/write, it is about 5-10 times faster than parquet writing or pystore
For append, it is marginally slower than pystore 

```
import numpy as np; import pandas as pd
from pyg_npy import pd_to_npy, pd_read_npy
import pystore
import datetime

pystore.set_path("c:/temp/pystore")
store = pystore.store('mydatastore')
collection = store.collection('NASDAQ')
arr = np.random.normal(0,1,(100,10))
df = pd.DataFrame(arr, columns = list('abcdefghij'))
dates = [datetime.datetime(2020,1,1) + datetime.timedelta(i) for i in range(-10000,0)]
ts = pd.DataFrame(np.random.normal(0,1,(10000,10)), dates, columns = list('abcdefghij'))

### write
%timeit collection.write('TEST', df, overwrite = True)
19.5 ms ± 1.97 ms per loop (mean ± std. dev. of 7 runs, 100 loops each)

%timeit df.to_parquet('c:/temp/test.parquet')
9.53 ms ± 650 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)

%timeit pd_to_npy(df, 'c:/temp/test.npy')
947 µs ± 38.5 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)


### read
%timeit collection.item('TEST').to_pandas()
7.7 ms ± 171 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)

%timeit pd.read_parquet('c:/temp/test.parquet')
2.85 ms ± 54.2 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)

%timeit pd_read_npy('c:/temp/test.npy')
847 µs ± 39.8 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)

### append
# because we need to append data with increasing index value, we use this:

collection.write('TEST2', ts.iloc[:100], overwrite = True)
%time len([collection.append('TEST2', ts.iloc[i*100: i*100+100], npartitions=2) for i in range(1,100)])
Wall time: 12.1 s

pd_to_npy(ts.iloc[:100], 'c:/temp/test.npy')
%time len([pd_to_npy(ts.iloc[i*100: i*100+100], 'c:/temp/test.npy', 'a', True) for i in range(1,100)])
Wall time: 2.14 s

```
