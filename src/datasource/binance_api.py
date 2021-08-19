import re
from typing import Tuple, Optional

import matplotlib.pyplot as plt
import ta
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np
from portion import Interval
from sklearn.preprocessing import StandardScaler
from collections import Counter
import seaborn as sns
from binance.client import Client
from binance.helpers import dateparser
import portion as P
import pathlib


def select_from_df(df, interval: Interval, return_idx=False):
    assert interval.atomic
    idx = np.ones(len(df), dtype=bool)

    # limit left
    if interval.left is P.OPEN:
        idx &= df.index > interval.lower
    else:
        idx &= df.index >= interval.lower

    # limit right
    if interval.right is P.OPEN:
        idx &= df.index < interval.upper
    else:
        idx &= df.index <= interval.upper

    if return_idx:
        return idx
    else:
        df = df.copy()
        return df.iloc[idx]


class BinanceLines:
    def __init__(self, symbol, step, root):
        root = pathlib.Path(root)
        assert root.exists()
        self.dir = root / symbol
        self.symbol = symbol
        self.root = root
        self.dir.mkdir(exist_ok=True)
        self.step = step

    @property
    def filename(self):
        return self.dir / f'{self.symbol}-{self.step}.csv'

    def read_file(self) -> Optional[pd.DataFrame]:
        if self.filename.is_file():
            date_cols = ['dateTime', 'closeTime']
            df = pd.read_csv(self.filename, parse_dates=date_cols)
            for col in date_cols:
                df[col] = df[col].dt.tz_localize('utc')

            df.set_index('dateTime', inplace=True)
            return df
        else:
            return None

    def load_data(self, interval: 'Interval[datetime]') -> pd.DataFrame:
        orig_df = self.read_file()

        # get what is stored
        if orig_df is None:
            interval_stored = P.empty()
            orig_df = pd.DataFrame()
        else:
            interval_stored = P.closed(orig_df.index[0], orig_df.index[-1])

        # assert what is stored is atomic
        assert interval_stored.atomic

        # assert interval is atomic
        assert interval.atomic

        # get enclosure of what is stored and what is requested
        interval_enclosure = (interval_stored | interval).enclosure

        # download enclosure (minus) what is stored
        interval_to_download = interval_enclosure - interval_stored
        print('interval enclosure:', interval_enclosure)
        print('interval to download:', interval_to_download)

        if not interval_to_download.empty:
            downloaded_df = download(self.symbol, interval_to_download.lower, interval_to_download.upper, self.step)

            # add to original dataframe
            orig_df = orig_df.append(downloaded_df)

        # remove possible duplicate timestamps and sort by index
        orig_df.sort_index(inplace=True)
        orig_df = orig_df[~orig_df.index.duplicated(keep='first')]

        # save the dataframe
        if not interval_to_download.empty:
            self.save(orig_df)

        # select the requested interval and return it
        df = select_from_df(orig_df, interval)
        return df

    def save(self, df: pd.DataFrame):
        df.to_csv(self.filename, date_format='%Y-%m-%d %H:%M:%S')


def download(symbol, since: datetime, until: datetime, interval=Client.KLINE_INTERVAL_1MINUTE) -> pd.DataFrame:
    client = Client()  # you can pass your api key and secret, but it still works without it
    start_str = int(since.timestamp()*1000)
    end_str = int(until.timestamp()*1000)
    candles = client.get_historical_klines(symbol, interval, start_str, end_str, limit=1000)

    columns = ['dateTime', 'Open', 'High', 'Low', 'Close', 'Volume', 'closeTime', 'quoteAssetVolume',
               'numberOfTrades', 'takerBuyBaseVol', 'takerBuyQuoteVol', 'ignore']

    df = pd.DataFrame(candles, columns=columns)
    df.dateTime = pd.to_datetime(df.dateTime, unit='ms', utc=True)
    df.closeTime = pd.to_datetime(df.closeTime, unit='ms', utc=True)
    df.set_index('dateTime', inplace=True)

    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        df[col] = df[col].astype('float32')

    return df


if __name__ == '__main__':
    step = '1m'
    bl = BinanceLines('XRPUSDT',  step=step, root='../datas/binance',)
    since = dateparser.parse('90 days ago utc+3')
    until = dateparser.parse('now utc+3')
    interval = P.closed(since, until)
    df = bl.load_data(interval)
    print(df)
    print(df.describe())
