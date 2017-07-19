from pyalgotrade import strategy
from pyalgotrade.barfeed import googlefeed, yahoofeed
import datetime

import pandas_datareader.data as web


class MyStrategy(strategy.BacktestingStrategy):
    def __init__(self, feed, instrument):
        super(MyStrategy, self).__init__(feed)
        self.__instrument = instrument

    def onBars(self, bars):
        bar = bars[self.__instrument]
        self.info(bar)


feed = googlefeed.Feed()
feed.addBarsFromCSV("orcl", "orcl.csv")

myStrategy = MyStrategy(feed, "orcl")
myStrategy.run()
