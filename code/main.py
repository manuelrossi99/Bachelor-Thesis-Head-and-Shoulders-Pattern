
from collections import defaultdict


import numpy as np
import pandas as pd
from scipy.signal import argrelextrema
from statsmodels.nonparametric.kernel_regression import KernelReg
from pandas_datareader import data as pdr

data = pdr.DataReader('BTC-USD', 'yahoo', '2014-10-17', '2022-05-09')['Adj Close']

def find_extrema(s, bw='cv_ls'): #'cv_ls'
    """
    Input:
        s: prices as pd.series
        bw: bandwith as str or array like
    Returns:
        prices: with 0-based index as pd.series
        extrema: extrema of prices as pd.series
        smoothed_prices: smoothed prices using kernel regression as pd.series
        smoothed_extrema: extrema of smoothed_prices as pd.series
    """
    # Copy series so we can replace index and perform non-parametric
    # kernel regression.
    prices = s.copy()
    prices = prices.reset_index()
    prices.columns = ['date', 'price']
    prices = prices['price']

    kr = KernelReg(
        [prices.values],
        [prices.index.to_numpy()],
        var_type='c', bw=bw
    )
    f = kr.fit([prices.index])

    # Use smoothed prices to determine local minima and maxima
    smooth_prices = pd.Series(data=f[0], index=prices.index)
    smooth_local_max = argrelextrema(smooth_prices.values, np.greater)[0]
    smooth_local_min = argrelextrema(smooth_prices.values, np.less)[0]
    local_max_min = np.sort(
        np.concatenate([smooth_local_max, smooth_local_min]))
    smooth_extrema = smooth_prices.loc[local_max_min]

    # Iterate over extrema arrays returning datetime of passed
    # prices array. Uses idxmax and idxmin to window for local extrema.
    price_local_max_dt = []
    for i in smooth_local_max:
        if (i > 1) and (i < len(prices)-1):
            price_local_max_dt.append(prices.iloc[i-2:i+2].idxmax())

    price_local_min_dt = []
    for i in smooth_local_min:
        if (i > 1) and (i < len(prices)-1):
            price_local_min_dt.append(prices.iloc[i-2:i+2].idxmin())

    maxima = pd.Series(prices.loc[price_local_max_dt])
    minima = pd.Series(prices.loc[price_local_min_dt])
    extrema = pd.concat([maxima, minima]).sort_index()

    # Return series for each with bar as index
    return extrema, prices, smooth_extrema, smooth_prices






def find_patterns(extrema, max_bars=30):
    """
    Input:
        extrema: extrema as pd.series with bar number as index
        max_bars: max bars for pattern to play out
    Returns:
        patterns: patterns as a defaultdict list of tuples
        containing the start and end bar of the pattern
    """
    patterns = defaultdict(list)

    # Need to start at five extrema for pattern generation
    for i in range(5, len(extrema)):
        window = extrema.iloc[i-5:i]

        # A pattern must play out within max_bars (default 35)
        if (window.index[-1] - window.index[0]) > max_bars:
            continue

        # Using the notation from the paper to avoid mistakes
        e1 = window.iloc[0]
        e2 = window.iloc[1]
        e3 = window.iloc[2]
        e4 = window.iloc[3]
        e5 = window.iloc[4]


        # Head and Shoulders
        if (e1 > e2) and (e3 > e1) and (e3 > e5) and \
                (abs(e1 - e5) <= 0.04*np.mean([e1, e5])) and \
                (abs(e2 - e4) <= 0.04*np.mean([e2, e4])) and \
                (((e1 - e2) + (e5 - e4)) / (e3 - (e2 + e4) / 2) <= 0.7) and \
                (((e1 - e2) + (e5 - e4)) / (e3 - (e2 + e4) / 2) >= 0.25) and \
                (((e3 - (e2 + e4) / 2) / e3) >= 0.03):
            patterns['HS'].append((e1,e2,e3,e4,e5,window.index[1], window.index[3]))


        #Inverse Head and Shoulders
        elif (e1 < e2) and (e3 < e1) and (e3 < e5) and \
                (abs(e1 - e5) <= 0.04*np.mean([e1, e5])) and \
                (abs(e2 - e4) <= 0.04*np.mean([e2, e4])) and \
                (abs(((e1 - e2) + (e5 - e4)) / (e3 - (e2 + e4) / 2)) <= 0.7) and \
                (abs(((e1 - e2) + (e5 - e4)) / (e3 - (e2 + e4) / 2)) >= 0.25) and \
                (abs(((e3 - (e2 + e4) / 2) / e3)) >= 0.03):
            patterns['IHS'].append((window.index[0], window.index[-1]))




    return patterns

'''def profiths(data, douple):
    i = douple[0] + 1
    j = douple[0] + 3
    a = abs(data[i+1] - g(i, j, i+1 , data))
    b = abs(data[j+1] - g(i, j, j+1 , data))
    if (g(i, j, j+2 , data) > data[j+2]): #breaks neckline
        k= j+2
        while((abs(g(i,j,j+2,data)- data[k]) < a and g(i,j,j+2,data)- data[k] > 0) or (abs((g(i,j,j+2,data)- data[k])) < b and g(i,j,j+2,data)- data[k] < 0)):
            k= k+1
        return  1- data[k]/data[j+2]   , k
    else:
        return 0
'''

def g(x, points): # gerade durch e2 e4 #points = 'hs'[1]
    k = (points[3] - points[1])/ (points[6] - points[5])
    d = (points[3] - k * points[6])
    return k * x +d

def profiths(data, points):
    i = points[5]
    j = points[6]
    a = abs(points[2] - g(i+1 , points)) # takeprofit
    b = abs(points[4] - g(j+1 , points)) # stop loss
    k= j+2
    if (g(j+2 , points) > data[j+3]): #breaks neckline
        k= j+2
        while((abs(g(j+2,points)- data[k]) < a and g(j+2,points)- data[k] > 0) or (abs((g(j+2,points)- data[k])) <= b and g(j+2,points)- data[k] <= 0)):
            k= k+1
        return  1- data[k+1]/data[j+2]   #, k, a, b, data[k+1], data[points[5]], data[points[6]], g(i+1, points),data[j+2]
    else:
        return 0

def profitihs(data, points):
    i = points[5]
    j = points[6]
    a = abs(points[2] - g(i+1 , points)) # takeprofit
    b = abs(points[4] - g(j+1 , points)) # stop loss
    k= j+2
    if (g(j+2 , points) < data[j+3]): #breaks neckline
        k= j+2
        while((abs(g(j+2,points)- data[k]) < a and g(j+2,points)- data[k] < 0) or (abs((g(j+2,points)- data[k])) <= b and g(j+2,points)- data[k] >= 0)):
            k= k+1
        return  data[k+1]/data[j+2]   #, k, a, b, data[k+1], data[points[5]], data[points[6]], g(i+1, points),data[j+2]
    else:
        return 0

def main(data):
    a = find_extrema(data)[2]
    b = find_patterns(a)
    l = []
    l1=[]
    for p in range(len(b['HS'])):
        print(profiths(find_extrema(data)[3],b['HS'][p]))
        l.append(profiths(find_extrema(data)[3],b['HS'][p]))
    for p in range(len(b['IHS'])):
        print(profitihs(find_extrema(data)[3],b['IHS'][p]))
        l1.append(profiths(find_extrema(data)[3],b['IHS'][p]))
    return l


main(data)

#a=[500,400,900,450,500,200,100,1,1,1]
#print(profiths(a,(0,4)))
#print(data[151:165])

###a = find_extrema(data)[2]
###b = find_patterns(a)
###print(profiths(find_extrema(data)[3],b['HS'][1]))

#print(find_extrema(data)[2])
##print(find_patterns(a))
#print(profiths(find_extrema(data)[3],find_patterns(find_extrema(a)['HS'])))
###print(find_patterns(a)['HS'][1][5])
#print(find_extrema(data)[3][])
#[473:478]
#b = find_extrema(data)[2]
#print(data[503:508])
#376.6199951171875
#478 376.6199951171875


#a = find_extrema(data)[3]
#b= find_extrema(data)[3]
#print(find_patterns(a))
#print(find_patterns(b))



