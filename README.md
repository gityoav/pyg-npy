# pyg-npy

pip install from https://pypi.org/project/pyg-npy/

A quick utility to save dataframes as npy files. 
It supports append and checks lightly on column names matching and index. 

For simple read/write, it is about 5-10 times faster than parquet writing or pystore
For append, it is marginally slower than pystore 

```
import numpy as np; import pandas as pd
from pyg_npy import pd_to_npy, pd_read_npy
import pystore
pystore.set_path("c:/temp/pystore")
store = pystore.store('mydatastore')
collection = store.collection('NASDAQ')
arr = np.random.normal(0,1,(100,10))
df = pd.DataFrame(arr, columns = list('abcdefghij'))


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
%timeit pd_to_npy(df, 'c:/temp/test.npy', mode = 'a', check = False)
12.7 ms ± 467 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)

```