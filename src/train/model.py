import time

import pandas as pd
import datetime
import pathlib
import pickle
from collections import Counter
from functools import partial
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

import numpy as np
import portion as P
from catboost import CatBoostClassifier, CatBoostRegressor
from scipy.optimize import brute
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

from datasource.binance_api import BinanceLines
from train.preprocessing import load_config, Pipeline
from train.rough_backtest import backtest_regressor


def rough_score(cfg, y_pred, y_true):
    # consider buying when y_pred says so
    # count how many wins and losses would there be
    idx = y_pred == 1  # select indices where prediction is to buy

    # the compute how much win and loss
    wins = sum(y_true[idx] == 1)
    losses = sum(y_true[idx] == -1)
    neutrals = sum(y_true[idx] == 0)

    score = wins * cfg['percent_increase'] + losses * (-cfg['percent_increase']) + (wins + losses + neutrals) * (
        -0.002)  # -0.001 is Binance's trading fee, so to buy and sell it takes 0.002

    return score, wins, losses, neutrals


def score_func(cfg, args, y_prob, y_pred, y_true, return_all=False):
    alpha, beta = args
    confident_indices = np.ones(len(y_prob), dtype=bool)
    confident_indices &= y_prob[:, 0] < y_prob[:, 1] * alpha
    confident_indices &= y_prob[:, 1] < y_prob[:, 2] * beta

    res = rough_score(cfg, y_pred[confident_indices], y_true[confident_indices])
    if return_all:
        return res
    else:
        return -res[0]


def report_classification(clf, x, y, alpha=None, beta=None):
    y_true = y
    y_pred = clf.predict(x).ravel()
    y_prob = clf.predict_proba(x)
    print(classification_report(y_pred, y_true))
    print('confusion matrix:')
    print(confusion_matrix(y_true, y_pred))

    # initial rough score
    # print rough score, wins, losses, neutral trades
    print('initial rough score:')
    print(rough_score(cfg, y_pred, y_true))

    score_func_ = partial(score_func, cfg)

    if alpha is None:
        alpha, beta = brute(score_func_, ranges=[(0, 1), (0, 1)], args=(y_prob, y_pred, y_true, False), Ns=20)
    print('optimal rough score:')
    print(score_func(cfg, (alpha, beta), y_prob, y_pred, y_true, return_all=True))

    print('alpha, beta:', (alpha, beta))
    return {'alpha': alpha, 'beta': beta}


class ClassifierModel:
    def __init__(self, cfg):
        self.clf = CatBoostClassifier(**cfg['classifier_params'])

    def train(self, splits):
        clf = self.clf
        train_x, train_y = splits['train']['X'], splits['train']['y']
        validation_x, validation_y = splits['validation']['X'], splits['validation']['y']
        test_x, test_y = splits['test']['X'], splits['test']['y']

        train_x = np.stack(train_x.values)
        validation_x = np.stack(validation_x.values)
        test_x = np.stack(test_x.values)

        clf.fit(train_x, train_y,
                eval_set=(validation_x, validation_y)
                )

        print('validation report:')
        val_result = report_classification(clf, validation_x, validation_y,
                                           # alpha=0.45, beta=1.
                                           )
        #
        # print('test report:')
        # test_result = report(clf, test_x, test_y,
        #                      alpha=0.45, beta=0.9
        #                      )


class RegressionModel:
    def __init__(self, cfg, log=True):
        self.clf = CatBoostRegressor(**cfg['regressor_params'])
        self.cfg = cfg

        self.log = log
        self.create_log_file()

    def create_log_file(self, suffix=''):
        if self.log:
            self.log_file = pathlib.Path('../logs/regression_logs/{}{}.txt'.format(
                time.strftime("%Y%m%d-%H%M%S"), suffix
            ))
            self.log_file.parent.mkdir(exist_ok=True)
            self.log_file = self.log_file.open('w')

    def __getstate__(self):
        return dict((k, v) for k, v in self.__dict__.items()
                    if not k in ['log_file'])

    def __setstate__(self, state):
        for k, v in state.items():
            self.__setattr__(k, v)
        self.create_log_file()

    def predict(self, X: pd.Series):
        if self.log:
            ys = []
            for time, x in X.iteritems():
                ys.append(self.predict_sample(x, time))
            return np.array(ys)
        else:
            xs = np.stack(X.values)
            return self.clf.predict(xs).ravel()

    def predict_sample(self, x: np.ndarray, time: datetime.datetime):
        assert x.ndim == 1
        y = self.clf.predict(x.reshape(1, -1)).ravel()[0]

        if self.log:
            self.log_file.write(
                f'time: {time}  |  prediction: {y} \n{x}\n\n'
            )
        return y

    def train(self, splits):
        clf = self.clf
        train_x, train_y = splits['train']['X'], splits['train']['y']
        validation_x, validation_y = splits['validation']['X'], splits['validation']['y']
        test_x, test_y = splits['test']['X'], splits['test']['y']

        train_x = np.stack(train_x.values)
        validation_x = np.stack(validation_x.values)

        clf.fit(train_x, train_y,
                eval_set=(validation_x, validation_y)
                )

        backtest_regressor(cfg, self, splits['validation']['X'], validation_y)
        backtest_regressor(cfg, self, splits['test']['X'], test_y)


def print_stats(splits):
    train_y = splits['train']['y']
    validation_y = splits['validation']['y']
    test_y = splits['test']['y']

    fig, axes = plt.subplots(1, 3)

    axes[0].hist(train_y, label='train', bins=50)
    axes[0].set_title('"Buy" points distribution for training set')

    axes[1].hist(validation_y, label='val', bins=50)
    axes[1].set_title('"Buy" points distribution for validation set')

    axes[2].hist(test_y, label='test', bins=50)
    axes[2].set_title('"Buy" points distribution for testing set')

    for ax in axes:
        ax.set(xlim=(-0.01, 0.01))

    fig.set_size_inches(12, 4)
    plt.tight_layout()
    plt.show()
    #
    # print('train_y distribution:', plt.hist(train_y, ))
    # print('validation_y distribution:', plt.hist(validation_y, label='validation', bins=50))
    # print('test_y distribution:', plt.hist(test_y, label='test', bins=50))
    # plt.show()


def save(obj, filename):
    pickle.dump(obj, open(filename, 'wb'))


def load(filename):
    return pickle.load(open(filename, 'rb'))


def plot_predictions(pipeline, model, df):
    cfg = pipeline.cfg
    Xy = pipeline.transform(df)

    a = cfg['train_until_date'] - datetime.timedelta(minutes=155)
    b = cfg['test_until_date']

    Xportion = Xy['X'][a:b]
    y_pred = pd.Series(model.predict(Xportion), index=Xportion.index)

    # select validate test region
    df = df[a - datetime.timedelta(hours=8):b]

    idx = y_pred.index[y_pred > 0.004]

    # plot candlesticks
    fig = go.Figure(go.Candlestick(x=df.index,
                                   open=df['Open'],
                                   high=df['High'],
                                   low=df['Low'],
                                   close=df['Close'],
                                   name='candles'))

    mean_price_indc = cfg['mean_price_indc']
    # add buy triggers
    fig.add_trace(go.Scatter(mode="markers", x=idx, y=df['Close'][idx],
                             marker_symbol='triangle-up',
                             marker_line_color="midnightblue", marker_color="red",
                             marker_line_width=1, marker_size=10,
                             # hovertemplate="name: %{y}%{x}<br>number: %{marker.symbol}<extra></extra>"
                             ))

    # draw mean price indicator
    fig.add_trace(go.Scatter(x=df.index, y=df[mean_price_indc], name=mean_price_indc))

    fig.update_layout(xaxis_rangeslider_visible=False)

    fig.write_html('fig.html')
    # fig.show()


if __name__ == '__main__':
    root_dir = pathlib.Path('../')

    cfg = load_config(root_dir / 'config/config_preprocess.yaml')
    print(cfg)

    # load data
    bl = BinanceLines(cfg['symbol'], cfg['step'], root_dir / 'datas/binance')
    interval = P.closed(cfg['since_date'], cfg['test_until_date'])
    df = bl.load_data(interval)

    # create data pipepline
    data_pipeline = Pipeline(cfg)
    splits = data_pipeline.fit(df)
    # splits = data_pipeline.transform(df)

    print_stats(splits)

    if cfg['model'] == 'regression':
        model = RegressionModel(cfg)
    else:
        model = ClassifierModel(cfg)

    model.train(splits)

    # save models
    save({'model': model,
          'pipeline': data_pipeline},
         '../dumps/model.pkl')

    import pprint

    pprint.pprint(cfg)

    df = bl.load_data(interval)
    model.create_log_file('_model_final')
    plot_predictions(data_pipeline, model, df)
