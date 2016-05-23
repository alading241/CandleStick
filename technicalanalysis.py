#!/usr/bin/env python

import sys
import datetime
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter, WeekdayLocator, MonthLocator, DayLocator
from matplotlib.finance import fetch_historical_yahoo, candlestick_ochl, candlestick2_ohlc, volume_overlay
from matplotlib import gridspec
from matplotlib.mlab import csv2rec
from matplotlib.dates import num2date, date2num, IndexDateFormatter
from matplotlib.ticker import  IndexLocator, FuncFormatter
import bisect

from dateutil.parser import parse

from operator import itemgetter

def get_locator():
    """
    the axes cannot share the same locator, so this is a helper
    function to generate locators that have identical functionality
    """
    return IndexLocator(10, 0.5)

def millions(x, pos):
    'The two args are the value and tick position'
    return '%1.1fM' % (x*1e-6)

def thousands(x, pos):
    'The two args are the value and tick position'
    return '%1.1fK' % (x*1e-3)

def MA(x, n, type='simple'):
    """
    compute an n period moving average.

    type is 'simple' | 'exponential'

    """
    x = np.asarray(x)
    if type=='simple':
        weights = np.ones(n)
    else:
        weights = np.exp(np.linspace(-1., 0., n))

    weights /= weights.sum()


    a =  np.convolve(x, weights, mode='full')[:len(x)]
    a[:n] = a[n]
    return a

"""
def SMA(x, n):
    x = np.asarray(x)
    a = np.ones(len(x))
    i = len(x) - 1
    while i > 0:
        s = i - n
        if s < 0: s = 0
        a[i]=x[s:i].mean()
        i = i - 1
    a[i]=x[i]
    return a
"""
def SMA(x,n):
    x = np.asarray(x)
    m = len(x)
    a = np.ones(m)

    for i in range(n-1,m):
        a[i] = x[i-n+1:i+1].mean()

    a[:n]=np.NaN
    return a

def DonchianHi(x, n):
    x = np.asarray(x)
    a = np.ones(len(x))
    i = len(x) - 1
    while i > 0:
        s = i - n
        if s < 0: s = 0
        a[i]=x[s:i].max()
        i = i - 1
    a[i]=x[i]
    return a

def DonchianLo(x, n):
    x = np.asarray(x)
    a = np.ones(len(x))
    i = len(x) - 1
    while i > 0:
        s = i - n
        if s < 0: s = 0
        a[i]=x[s:i].min()
        i = i - 1
    a[i]=x[i]
    return a

def BB(x, n, nSTD=2.0):

    m = len(x)

    if m < n:
        # show error message
        raise SystemExit('Error: num_prices < period')

    # 3 bands, bandwidth, range and %B
    bbs = np.zeros((m, 6))

    ma = SMA(x, n)

    for i in range(n-1, m):
        std = np.std(x[i-n+1:i+1])

        # upper, middle, lower bands, bandwidth, range and %B
        bbs[i, 0] = ma[i] + std * nSTD
        bbs[i, 1] = ma[i]
        bbs[i, 2] = ma[i] - std * nSTD
        bbs[i, 3] = (bbs[i, 0] - bbs[i, 2]) / bbs[i, 1]
        bbs[i, 4] = bbs[i, 0] - bbs[i, 2]
        bbs[i, 5] = (x[i] - bbs[i, 2]) / bbs[i, 4] if bbs[i,4]!=0 else np.NaN

    bbs[:n, 0] = np.NaN
    bbs[:n, 1] = np.NaN
    bbs[:n, 2] = np.NaN
    bbs[:n, 3] = np.NaN
    bbs[:n, 4] = np.NaN
    bbs[:n, 5] = np.NaN

    return bbs

def bb(prices, period, num_std_dev=2.0):

    num_prices = len(prices)

    if num_prices < period:
        # show error message
        raise SystemExit('Error: num_prices < period')

    # 3 bands, bandwidth, range and %B
    bbs = np.zeros((num_prices, 6))

    ma = SMA(prices, period)

    idx = num_prices - 1
    while idx > 0:
        s = idx - period
        if s < 0: s=0
        std_dev = np.std(prices[s:idx])

        # upper, middle, lower bands, bandwidth, range and %B
        bbs[idx, 0] = ma[idx] + std_dev * num_std_dev
        bbs[idx, 1] = ma[idx]
        bbs[idx, 2] = ma[idx] - std_dev * num_std_dev
        bbs[idx, 3] = (bbs[idx, 0] - bbs[idx, 2]) / bbs[idx, 1]
        bbs[idx, 4] = bbs[idx, 0] - bbs[idx, 2]
        bbs[idx, 5] = (prices[idx] - bbs[idx, 2]) / bbs[idx, 4] if bbs[idx,4]!=0 else np.NaN

    bbs[idx, 0] = np.NaN
    bbs[idx, 1] = np.NaN
    bbs[idx, 2] = np.NaN
    bbs[idx, 3] = np.NaN
    bbs[idx, 4] = np.NaN
    bbs[idx, 5] = np.NaN

    return bbs

def RSI(prices, n=14):
    """
    compute the n period relative strength indicator
    http://stockcharts.com/school/doku.php?id=chart_school:glossary_r#relativestrengthindex
    http://www.investopedia.com/terms/r/rsi.asp
    """

    deltas = np.diff(prices)
    seed = deltas[:n+1]
    up = seed[seed>=0].sum()/n
    down = -seed[seed<0].sum()/n
    rs = up/down
    rsi = np.zeros_like(prices)
    rsi[:n] = 100. - 100./(1.+rs)

    for i in range(n, len(prices)):
        delta = deltas[i-1] # cause the diff is 1 shorter

        if delta>0:
            upval = delta
            downval = 0.
        else:
            upval = 0.
            downval = -delta

        up = (up*(n-1) + upval)/n
        down = (down*(n-1) + downval)/n

        rs = up/down
        rsi[i] = 100. - 100./(1.+rs)

    return rsi

def MACD(x, nslow=26, nfast=12):
    """
    compute the MACD (Moving Average Convergence/Divergence) using a fast and slow exponential moving avg'
    return value is emaslow, emafast, macd which are len(x) arrays
    """
    emaslow = moving_average(x, nslow, type='exponential')
    emafast = moving_average(x, nfast, type='exponential')
    return emaslow, emafast, emafast - emaslow

def getListOfDates(startdate, enddate):
    "Need to modify the function name"
    dates = [datetime.date(m/12, m%12+1, 1) for m in range(startdate.year*12+startdate.month-1, enddate.year*12+enddate.month)]
    return np.array(dates)

def getDateIndex(dates, tickdates):
    "Need to modify the function name"
    index = [bisect.bisect_left(dates, tickdate) for tickdate in tickdates]
    return np.array(index)

def getMonthNames(dates, index):
    names = [dates[i].strftime("%b'%y") for i in index]
    return np.array(names)

def format_coord1(x, y):
    return 'x=%s, y=%1.1f' % (r.date[x+0.5], y)
    #return 'x=%s, y=%s' % (x,y)

def format_coord2(x, y):
    return 'x=%s, y=%1.1fM' % (r.date[x+0.5], y*1e-6)


if __name__=="__main__":

    if len(sys.argv)<3:
        print u"Usage: python technicalanalysis.py ticker startdate [enddate]"
        raise SystemExit

    ticker = sys.argv[1]
    startdate = parse(sys.argv[2])

    if len(sys.argv)==4:
        enddate = sys.argv[3]
    else:
        enddate = datetime.date.today()

    fh = fetch_historical_yahoo(ticker, startdate, enddate)
    # a numpy record array with fields: date, open, high, low, close, volume, adj_close)

    r = csv2rec(fh); fh.close()
    r.sort()


    if len(r.date) == 0:
        raise SystemExit

    tickdates = getListOfDates(startdate, enddate)
    tickindex = getDateIndex(r.date, tickdates)
    ticknames = getMonthNames(r.date, tickindex)

    formatter =  IndexDateFormatter(date2num(r.date), '%m/%d/%y')

    millionformatter = FuncFormatter(millions)
    thousandformatter = FuncFormatter(thousands)

    fig = plt.figure()
    fig.subplots_adjust(bottom=0.1)
    fig.subplots_adjust(hspace=0)

    gs = gridspec.GridSpec(2, 1, height_ratios=[4, 1])

    ax0 = plt.subplot(gs[0])

    candles = candlestick2_ohlc(ax0, r.open, r.high, r.low, r.close, width=1, colorup='g', colordown='r')

    ma05 = MA(r.adj_close, 5, type='simple')
    ma20 = MA(r.close, 20, type='simple')
    ma60 = MA(r.adj_close, 60, type='simple')

    Don20Hi = DonchianHi(r.high, 20)
    Don20Lo = DonchianLo(r.low, 20)

    BB20 = BB(r.close, 20)

    ma05[:5]=np.NaN
    ma20[:20]=np.NaN
    ma60[:60]=np.NaN

    ax0.plot(ma05, color='black', lw=2, label='MA (5)')
    ax0.plot(ma20, color='blue', lw=2, label='MA (20)')
    ax0.plot(ma60, color='red', lw=2, label='MA (60)')
    ax0.plot(Don20Hi, color='blue', lw=2, ls='--', label='DonHi (20)')
    ax0.plot(Don20Lo, color='blue', lw=2, ls='--', label='DonLo (20)')

    ax0.plot(range(len(r.date)), BB20[:,0], color='black', lw=2, ls='--', label='BBU (20)')
    ax0.plot(range(len(r.date)), BB20[:,1], color='black', lw=2, ls='--', label='BBM (20)')
    ax0.plot(range(len(r.date)), BB20[:,2], color='black', lw=2, ls='--', label='BBD (20)')
    ax0.fill_between(range(len(r.date)), BB20[:,0],BB20[:,2], facecolor='#cccccc', alpha=0.5)

    ax0.set_xticks(tickindex)
    ax0.set_xticklabels(ticknames)
    ax0.format_coord=format_coord1
    ax0.legend(loc='best', shadow=True, fancybox=True)
    ax0.set_ylabel('Price($)', fontsize=16)
    ax0.set_title(ticker, fontsize=24, fontweight='bold')
    ax0.grid(True)

    ax1 = plt.subplot(gs[1], sharex=ax0)

    vc = volume_overlay(ax1, r.open, r.close, r.volume, colorup='g', width=1)

    ax1.set_xticks(tickindex)
    ax1.set_xticklabels(ticknames)
    ax1.format_coord=format_coord2

    ax1.tick_params(axis='x',direction='out',length=5)
    ax1.yaxis.set_major_formatter(millionformatter)
    ax1.yaxis.tick_right()
    ax1.yaxis.set_label_position("right")
    ax1.set_ylabel('Volume', fontsize=16)
    ax1.grid(True)

    plt.setp(ax0.get_xticklabels(), visible=False)

    plt.show()
