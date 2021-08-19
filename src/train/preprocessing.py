import re
from abc import abstractmethod
from typing import Tuple, Optional, Dict, Union
import yaml
import joblib

import matplotlib.pyplot as plt
import pytz
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

from datasource.binance_api import BinanceLines


class Pipe:
    def __init__(self, func):
        self.func = func
        self.hooks = []

    def __call__(self, *args, **kwargs):
        output = self.func(*args, **kwargs)
        for hook in self.hooks:
            hook(output)
        return output

    def add_hook(self, hook):
        self.hooks = hook

    def remove_hook(self, hook):
        self.hooks.remove(hook)


def pipe(func):
    return Pipe(func)


@pipe
def add_indicators(cfg, df: pd.DataFrame, dropna=True):
    def parse_params(s):
        return [int(x) for x in s.split('_')[1:]]

    required_features = list(set(cfg['training_features'] + [cfg['mean_price_indc']])
                             - {'Open', 'Close', 'High', 'Low'}
                             )

    for feat in required_features:
        params = parse_params(feat)
        if feat.startswith('ma'):
            indc = ta.trend.SMAIndicator(df['Close'], *params).sma_indicator()
        elif feat.startswith('ppo'):
            indc = ta.momentum.PercentagePriceOscillator(df['Close'], *params).ppo()
        elif feat.startswith('ema'):
            indc = ta.trend.EMAIndicator(df['Close'], *params).ema_indicator()
        elif feat == 'adi':
            indc = ta.volume.AccDistIndexIndicator(df['High'], df['Low'], df['Close'], df['Volume']).acc_dist_index()
        else:
            raise ValueError(f'Please implement the parsing of this indicator: {feat}')
        df[feat] = indc

    if dropna:
        df.dropna(inplace=True)
    return df

    #
    # df['ma2'] = ta.trend.SMAIndicator(df['Close'], 2).sma_indicator()
    # df['ma3'] = ta.trend.SMAIndicator(df['Close'], 3).sma_indicator()
    # df['ma4'] = ta.trend.SMAIndicator(df['Close'], 4).sma_indicator()
    # df['ma5'] = ta.trend.SMAIndicator(df['Close'], 5).sma_indicator()
    # df['ma7'] = ta.trend.SMAIndicator(df['Close'], 7).sma_indicator()
    # df['ma9'] = ta.trend.SMAIndicator(df['Close'], 9).sma_indicator()
    # df['ma13'] = ta.trend.SMAIndicator(df['Close'], 13).sma_indicator()
    #
    # # df['ema5'] = ta.trend.EMAIndicator(df['Close'], 5).ema_indicator()
    # # df['ema7'] = ta.trend.EMAIndicator(df['Close'], 7).ema_indicator()
    # df['ppo13_5'] = ta.momentum.PercentagePriceOscillator(df['Close'], 13, 5).ppo()
    # df['ppo27_7'] = ta.momentum.PercentagePriceOscillator(df['Close'], 27, 7).ppo()
    # df['ppo27_5'] = ta.momentum.PercentagePriceOscillator(df['Close'], 27, 5).ppo()
    # df['ppo9_3'] = ta.momentum.PercentagePriceOscillator(df['Close'], 9, 3).ppo()
    # df['ppo20_2'] = ta.momentum.PercentagePriceOscillator(df['Close'], 20, 2).ppo()
    # df['ppo30_10'] = ta.momentum.PercentagePriceOscillator(df['Close'], 30, 10).ppo()
    #
    # # df['ppo25_7'] = ta.momentum.PercentagePriceOscillator(df['Close'], 25, 7).ppo()
    # # df['ppo25_5'] = ta.momentum.PercentagePriceOscillator(df['Close'], 25, 5).ppo()
    # df['ppo12_4'] = ta.momentum.PercentagePriceOscillator(df['Close'], 12, 4).ppo()
    # df['ppo24_4'] = ta.momentum.PercentagePriceOscillator(df['Close'], 24, 4).ppo()
    # df['ppo120_4'] = ta.momentum.PercentagePriceOscillator(df['Close'], 120, 4).ppo()
    # df['ppo120_7'] = ta.momentum.PercentagePriceOscillator(df['Close'], 120, 7).ppo()
    #
    # df['rsi13'] = ta.momentum.RSIIndicator(df['Close'], 13).rsi()
    # # df['stoch_rsi'] = ta.momentum.StochRSIIndicator(df['Close'], 13).stochrsi()
    # df['EoM'] = ta.volume.EaseOfMovementIndicator(df['High'], df['Low'], df['Close'], df['Volume']).ease_of_movement()
    # df['vpt'] = ta.volume.volume_price_trend(df['Close'], df['Volume'])
    # # df['pvo'] = ta.momentum.PercentageVolumeOscillator(volume=df['Volume']).pvo()
    # df['macd'] = ta.trend.MACD(close=df['Close']).macd()
    # # df['mi'] = ta.trend.MassIndex(high=df['High'], low=df['Low'], window_fast=7, window_slow=27).mass_index()


def compute_percentage_deviation(cfg, df):
    # TODO: you can only consider since 'since' for recency

    # get rolling windows of length 'target_future'
    roll = df[cfg['mean_price_indc']].rolling(cfg['target_future'])

    # compute std of price indicators, take x-quantile, normalize by mean mean
    return roll.std().quantile(cfg['price_deviation_std_quantile']) / roll.mean().mean()


@pipe
def add_buy_signal(cfg, df: pd.DataFrame):
    # define percentage increase as percentage deviation
    # if cfg['percent_increase'] is None:
    #     cfg['percent_increase'] = compute_percentage_deviation(cfg, df)
    pc = cfg['percent_increase']
    if pc < 0.003:
        raise ValueError(f'PERCENT INCREASE TOO LOW!  ({pc})')

    # compute "buy" column
    buys = {}

    # buy at the next open
    # buy_price = df['Open'].shift(-1).tolist()
    cur_price = df[cfg['mean_price_indc']].tolist()

    for i in range(len(df)):
        date = df.index[i]
        if i + cfg['target_future'] >= len(df):
            buys[date] = np.nan  # no future at this time
            continue

        fut_price = df[cfg['mean_price_indc']][i + 1:i + cfg['target_future'] + 1]

        if max(fut_price) / cur_price[i] > 1 + pc:
            b = 1
        elif min(fut_price) / cur_price[i] < 1 - pc:
            b = -1
        else:
            b = 0

        buys[date] = b
        #
        # fut_max = max(df['High'][i + 1:i + cfg['target_future'] + 1])
        # fut_close = df['Close'][i + cfg['target_future']]
        #
        # future_highs = df['High'][i + 1:i + cfg['target_future'] + 1]
        # future_lows = df['Low'][i + 1:i + cfg['target_future'] + 1]
        #
        # wins_vector = (future_highs >= buy_price[i] * (1 + pc)).tolist()
        # lose_vector = (future_lows <= buy_price[i] * (1 - pc)).tolist()
        #
        # if wins_vector > lose_vector:
        #     b = 1
        # elif wins_vector < lose_vector:
        #     b = -1
        # else:
        #     b = 0
        # buys[date] = b

    df['buy'] = pd.Series(buys)
    df.dropna(inplace=True)
    return df


@pipe
def add_percent_increase_column(cfg, df: pd.DataFrame, dropna=False):
    # How much will the moving average price will rise in the future n steps

    # compute "buy" column
    percent_increase = {}

    mean_prices = df[cfg['mean_price_indc']].tolist()

    for i in range(len(df)):
        date = df.index[i]
        if i + cfg['target_future'] >= len(df):
            percent_increase[date] = np.nan  # no future at this time
            continue

        cur_price = mean_prices[i]
        fut_price = mean_prices[i + cfg['target_future']]
        percent_increase[date] = (fut_price - cur_price) / cur_price

    df['percent_increase'] = pd.Series(percent_increase)
    if dropna:
        df = df.dropna()
    return df


def _create_windows_from_array(X, y, window_size):
    # given time series data X, and labels y, return samples
    # with previous timesteps of size `window_size` concatenated together
    # returns also indices of y which were included

    assert len(X) == len(y)

    result_X = []
    result_y = []
    indices = []

    for i in range(window_size - 1, X.shape[0]):
        result_X.append(X[i - window_size + 1:i + 1].ravel())
        result_y.append(y[i])
        indices.append(i)

    result_X = np.array(result_X)
    result_y = np.array(result_y)

    return result_X, result_y, indices


@pipe
def create_windows(cfg, df: pd.DataFrame) -> Dict[str, pd.Series]:
    # create training dataset, split into train, validation, and test
    X = df[cfg['training_features']].values
    y = df[cfg['target_column']].values

    X_windowed, y, _xy_indices = _create_windows_from_array(X, y, cfg['window_size'])
    xy_dates = df.index[_xy_indices]

    if len(xy_dates) == 0:
        raise NotEnoughData
    return {'X': pd.Series(list(X_windowed), index=xy_dates), 'y': pd.Series(y, index=xy_dates)}


@pipe
def train_val_test_split(Xy, train_until_date, validation_until_date) -> Dict[str, Dict[str, pd.Series]]:
    X, y = Xy['X'], Xy['y']
    initial_date = X.index[0]

    train_Xy, validation_Xy, test_Xy = {}, {}, {}
    train_Xy['X'] = X[initial_date:train_until_date]
    train_Xy['y'] = y[initial_date:train_until_date]
    validation_Xy['X'] = X[train_until_date:validation_until_date]
    validation_Xy['y'] = y[train_until_date:validation_until_date]
    test_Xy['X'] = X[validation_until_date:]
    test_Xy['y'] = y[validation_until_date:]

    return {'train': train_Xy, 'validation': validation_Xy, 'test': test_Xy}


class NotEnoughData(Exception):
    pass


@pipe
def normalize_data(splitted_data: Dict[str, Dict[str, pd.Series]] = None,
                   Xy: Dict[str, pd.Series] = None, scaler=None) \
        -> Union[Tuple[Dict[str, Dict[str, pd.Series]], StandardScaler],
                 Dict[str, pd.Series]]:
    if Xy is not None:
        assert scaler is not None

        if len(Xy['X']) == 0:
            raise NotEnoughData('not enough data')

        Xy = Xy.copy()
        Xy['X'].iloc[:] = list(scaler.transform(np.stack(Xy['X'])))
        return Xy

    train_x = np.stack(splitted_data['train']['X'].values)
    validation_x = np.stack(splitted_data['validation']['X'].values)
    test_x = np.stack(splitted_data['test']['X'].values)

    assert train_x.ndim == 2

    if scaler is None:
        # fit scaler on train and validation
        scaler = StandardScaler().fit(np.concatenate([train_x, validation_x], axis=0))

    train_x = scaler.transform(train_x)
    validation_x = scaler.transform(validation_x)
    test_x = scaler.transform(test_x)

    splitted_data['train']['X'].iloc[:] = list(train_x)
    splitted_data['validation']['X'].iloc[:] = list(validation_x)
    splitted_data['test']['X'].iloc[:] = list(test_x)

    return splitted_data, scaler


class Pipeline:
    cfg = None
    scaler = None

    def __init__(self, cfg):
        self.cfg = cfg

    def fit(self, df):
        cfg = self.cfg
        add_indicators(cfg, df)
        # add_buy_signal(cfg, df)
        add_percent_increase_column(cfg, df, dropna=True)
        Xy = create_windows(cfg, df)
        splitted_data = train_val_test_split(Xy, cfg['train_until_date'], cfg['validation_until_date'])

        if cfg['normalize_data']:
            splitted_data, self.scaler = normalize_data(splitted_data)
        return splitted_data

    def transform(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        cfg = self.cfg
        add_indicators(cfg, df)
        # add_buy_signal(cfg, df)
        add_percent_increase_column(cfg, df)
        Xy = create_windows(cfg, df)
        if cfg['normalize_data']:
            Xy = normalize_data(Xy=Xy, scaler=self.scaler)
        return Xy

    def __getstate__(self):
        return {'cfg': self.cfg, 'scaler': self.scaler}

    def __setstate__(self, state):
        self.cfg = state['cfg']
        self.scaler = state['scaler']


def parse_timedelta(str):
    str += ' ago'
    return datetime.now() - dateparser.parse(str)


def load_config(filename):
    cfg = yaml.safe_load(open(filename))

    def map_dates_to_pytz_utc(d):
        for k, v in sorted(d.items(), key=lambda x: x[0]):
            if isinstance(v, dict):
                map_dates_to_pytz_utc(v)
            elif isinstance(v, datetime):
                d[k] = v.astimezone(pytz.utc)

    map_dates_to_pytz_utc(cfg)

    cfg['since_date'] = cfg['train_until_date'] - parse_timedelta(cfg['train_duration'])
    cfg['validation_until_date'] = cfg['train_until_date'] + parse_timedelta(cfg['validation_duration'])
    cfg['test_until_date'] = cfg['validation_until_date'] + parse_timedelta(cfg['test_duration'])

    return cfg


if __name__ == '__main__':
    root_dir = pathlib.Path('../')

    cfg = load_config(root_dir / 'config/config_preprocess.yaml')

    # load data
    bl = BinanceLines('XRPUSDT', cfg['step'], root_dir / 'datas/binance')
    interval = P.closed(cfg['since_date'], cfg['test_until_date'])
    df = bl.load_data(interval)

    # create data pipepline
    data_pipeline = Pipeline(cfg)

    # fit pipeline
    data_pipeline.fit(df)

    # save pipeline
    joblib.dump(data_pipeline, root_dir / 'dumps/pipeline.joblib')

    data_pipeline.transform(df)
    import pprint

    pprint.pprint(cfg)
