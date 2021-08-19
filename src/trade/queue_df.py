import datetime

import dateparser
import pandas as pd


class TimeSeriesWindowDF:
    def __init__(self, max_length: datetime.timedelta, df: pd.DataFrame = None, index_key='date', dt_err=datetime.timedelta(seconds=10)):
        if df is not None:
            df = df.set_index(index_key)
        df = df if df is not None else pd.DataFrame()
        self.df = df
        self.max_length = max_length
        self.index_key = index_key
        self.dt_err = dt_err

    def get_length(self):
        if len(self.df) == 0:
            return 0
        return self.df.index[-1] - self.df.index[0]

    def pop(self):
        '''Pop earlies entered rows'''
        self.df.drop(self.df.index[0], inplace=True)

    def append(self, row):
        df = self.df
        new_df = pd.DataFrame([row])
        new_df.set_index(self.index_key, inplace=True)
        if len(df) > 0:
            assert row['date'] > df.index[-1]
        self.df = df.append(new_df, verify_integrity=True)
        while self.get_length() - self.max_length > self.dt_err:
            self.pop()

    def __repr__(self):
        return self.df.__repr__()


if __name__ == '__main__':
    minute = datetime.timedelta(minutes=1)
    dt = dateparser.parse('12:00')

    qdf = TimeSeriesWindowDF(datetime.timedelta(minutes=1))
    qdf.append(dict(date=dt, name='one', number=1))
    print(qdf.df)
    qdf.append(dict(date=dt + minute, name='two', number=2))
    print(qdf.df)
    qdf.append(dict(date=dt + minute + minute, name='three', number=3))
    print(qdf.df)
