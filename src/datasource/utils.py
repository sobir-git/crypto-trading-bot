from datetime import datetime
from typing import Iterable, Union

from pandas import DatetimeIndex
import pandas as pd

#
# class TimeSeries(pd.Series):
#     def __init__(self, values: Iterable, index: Union[pd.DatetimeIndex, datetime], *args, **kwargs):
#         values = list(values)
#         super().__init__(values, index, *args, **kwargs)
#
#     def get_values(self):
#         values = pd.Series.values.fget(self)
#         return np.stack(values, axis=0)


if __name__ == '__main__':
    import numpy as np

    values = np.arange(8).reshape(4, 2)
    index = DatetimeIndex(['2010', '2011', '2012', '2013'])

    ts = TimeSeries(values, index)
    print(ts['2010'])

    assert np.allclose(ts.values, values)
