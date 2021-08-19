import portion as P
import pathlib
import time
import backtrader as bt
import datetime as dt

import yaml
from ccxtbt import CCXTStore

from constants import *
from datasource.binance_api import BinanceLines
from trade.basic import CommentStrategy
from trade.generic_dataset import BinanceDataset
from trade.sizers import FullMoney
from train.model import RegressionModel
from train.preprocessing import Pipeline
from utils import print_trade_analysis, print_sqn, send_telegram_message


def main(cfg):
    cerebro = bt.Cerebro(quicknotify=True)

    if cfg['env'] == PRODUCTION:  # Live trading with Binance
        if cfg['binance'].get('key') is None:
            cfg['binance']['key'] = input('binance key: ')
            cfg['binance']['secret'] = input('binance secret: ')

        broker_config = {
            # 'urls': {'api': 'https://testnet.binance.vision'},
            'apiKey': cfg['binance'].get("key"),
            'secret': cfg['binance'].get("secret"),
            'nonce': lambda: str(int(time.time() * 1000)),
            'enableRateLimit': True,
        }

        store = CCXTStore(exchange='binance', currency=cfg['coin_refer'], config=broker_config, retries=5, debug=cfg['debug'])

        broker_mapping = {
            'order_types': {
                bt.Order.Market: 'market',
                bt.Order.Limit: 'limit',
                bt.Order.Stop: 'stop-loss',
                bt.Order.StopLimit: 'stop limit'
            },
            'mappings': {
                'closed_order': {
                    'key': 'status',
                    'value': 'closed'
                },
                'canceled_order': {
                    'key': 'status',
                    'value': 'canceled'
                }
            }
        }

        broker = store.getbroker(broker_mapping=broker_mapping)
        cerebro.setbroker(broker)

        hist_start_date = dt.datetime.utcnow() - dt.timedelta(minutes=5)
        data = store.getdata(
            dataname='%s/%s' % (cfg["coin_target"], cfg["coin_refer"]),
            name='%s%s' % (cfg["coin_target"], cfg["coin_refer"]),
            timeframe=bt.TimeFrame.Minutes,
            fromdate=hist_start_date,
            compression=1,
            ohlcv_limit=99999
        )

        # Add the feed
        cerebro.adddata(data)

    else:  # Backtesting with CSV file
        data = BinanceDataset(
            name=cfg["coin_target"],
            dataname=cfg["dataname"],
            timeframe=bt.TimeFrame.Minutes,
            fromdate=cfg["fromdate"],
            todate=cfg["todate"],
            nullvalue=0.0
        )

        cerebro.resampledata(data, timeframe=bt.TimeFrame.Minutes, compression=1)

        broker = cerebro.getbroker()
        broker.setcommission(commission=0.001, name=cfg["coin_target"])  # Simulating exchange fee
        broker.setcash(100000.0)
        cerebro.addsizer(FullMoney)

    # Analyzers to evaluate trades and strategies
    # SQN = Average( profit / risk ) / StdDev( profit / risk ) x SquareRoot( number of trades )
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="ta")
    cerebro.addanalyzer(bt.analyzers.SQN, _name="sqn")

    # temp
    root_dir = pathlib.Path('../')
    bl = BinanceLines('XRPUSDT', '1m', root_dir / 'datas/binance')
    interval = P.closed(cfg['fromdate'], cfg['todate'])
    df = bl.load_data(interval)

    # Include Strategy
    cerebro.addstrategy(CommentStrategy, cfg, df)

    # Starting backtrader bot
    initial_value = cerebro.broker.getvalue()
    print('Starting Portfolio Value: %.2f' % initial_value)
    result = cerebro.run()

    # Print analyzers - results
    final_value = cerebro.broker.getvalue()
    print('Final Portfolio Value: %.2f' % final_value)
    print('Profit %.3f%%' % ((final_value - initial_value) / initial_value * 100))
    print_trade_analysis(result[0].analyzers.ta.get_analysis())
    print_sqn(result[0].analyzers.sqn.get_analysis())

    if cfg['debug']:
        cerebro.plot()


if __name__ == "__main__":
    cfg = yaml.safe_load(open('../config/trade.yaml'))
    try:
        main(cfg)
    except KeyboardInterrupt:
        print("finished.")
        time = dt.datetime.now().strftime("%d-%m-%y %H:%M")
        send_telegram_message(cfg, "Bot finished by user at %s" % time)
    except Exception as err:
        send_telegram_message(cfg, "Bot finished with error: %s" % err)
        print("Finished with error: ", err)
        raise
