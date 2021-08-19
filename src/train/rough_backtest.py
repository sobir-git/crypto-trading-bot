import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def backtest_regressor(cfg, model, X, y):
    '''
    Say we want to use the regressor model which predicts how much percent will
    a specific MA indicator (e.g. MA(4)) will increase by the end of n future steps.

    One way to utilize this model is consider it's predictions when they are above a
    certain threshold, indicative of keeping long on the trade.

    Here we set the threshold to 0.005. We consider a commission of 0.001 for each trade (sell of buy).

    '''

    pred_y = model.predict(X).ravel()

    # filter only predictions where > p
    idx = np.nonzero(pred_y > 0.004)[0]

    # compute gain
    n_trades = len(idx)
    comm = 0.001
    gain = sum(y[idx]) - 2 * comm * n_trades

    print('number of trades:', n_trades)
    print('gain:', gain)

    sns.scatterplot(x=pred_y, y=y, hue=y > 2 * comm, palette={True: 'green', False: 'red'})
    plt.xlim((-0.01, 0.01))
    plt.ylim((-0.01, 0.01))
    plt.title(f'Gain: {gain}')
    plt.show()
