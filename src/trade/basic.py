import dateparser
import numpy as np
import datetime

import pandas as pd
from backtrader import num2date, Order

from trade.base import StrategyBase
from train.model import RegressionModel, load
from train.preprocessing import Pipeline, NotEnoughData

from constants import *


def construct_row_from_data(data):
    return {
        'date': data.datetime[0],
        'Open': data.open[0],
        'High': data.high[0],
        'Low': data.low[0],
        'Close': data.close[0],
        'Volume': data.volume[0],
    }


def construct_df_from_data(data, size=200):
    sz = min(len(data), 200)  # get last 150 points
    d = {
        'date': [data.num2date(x) for x in data.datetime.get(size=sz)],
        'Open': data.open.get(size=sz),
        'High': data.high.get(size=sz),
        'Low': data.low.get(size=sz),
        'Close': data.close.get(size=sz),
        'Volume': data.volume.get(size=sz),
    }
    df = pd.DataFrame(d)
    df.set_index('date', inplace=True)
    assert len(df) <= len(data)
    return df


class CommentStrategy(StrategyBase):
    params = dict(
        perc=0.003,
        risk_reward_ratio=1.5
    )

    def __init__(self, cfg, df):
        super(CommentStrategy, self).__init__(cfg)

        d = load(cfg['model'])

        self.pipeline: Pipeline = d['pipeline']
        self.model: RegressionModel = d['model']
        self.profit = 0
        self.timer_ = 0

        self.Xy = self.pipeline.transform(df)

    def next(self):
        cfg = self.cfg
        df = construct_df_from_data(self.data, 155)
        try:
            Xy_ = self.pipeline.transform(df.copy())
        except NotEnoughData:
            return

        # delay = num2date(self.data.datetime[0]) - Xy_['X'].index[-1]
        # if delay > datetime.timedelta(seconds=10):
        #     print('DELAY:', delay)

        if self.status != "LIVE" and cfg['env'] == PRODUCTION:  # waiting for live status in production
            print('waiting ... ', self.status)
            return

        # if last operation is buy, and current price has dropped below the threshold,
        # then cancel any pending order, and sell everything
        cur_close_price = self.data.close[0]
        if self.last_operation == BUY:
            if cur_close_price < self.sell_stop_loss_price:
                # cancel pending order
                self.broker.cancel(self.order)

                # sell quickly
                print('shorting because of loss')
                self.short()

        if self.order:  # waiting for pending order
            return

        x = Xy_['X'][-1]
        now = num2date(self.data.datetime[0])

        prediction = self.model.predict_sample(x, now)

        # if we haven't bought any
        if self.last_operation != BUY:
            # if we have buy signal
            if prediction > self.p.perc:
                print(f'{self.data.datetime.datetime()} | Place buy order because prediction is {prediction}')
                # immediately buy
                self.long()
                # start the countdown timer
                self.sell_profit_price = (1 + self.p.perc) * cur_close_price
                self.sell_stop_loss_price = (1 - self.p.perc * self.p.risk_reward_ratio) * cur_close_price
            return

    def notify_order(self, order):
        super(CommentStrategy, self).notify_order(order)

        if order.status in [order.Completed]:
            if order.isbuy():
                # sell short
                print(f'Sell short: limit={self.sell_profit_price}')
                self.sell(exectype=Order.Limit,
                          price=self.sell_profit_price)
