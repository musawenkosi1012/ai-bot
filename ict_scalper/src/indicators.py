# src/indicators.py
import numpy as np
import pandas as pd

def atr(series_high, series_low, series_close, period=14):
    """Return ATR series aligned with close"""
    high = np.asarray(series_high)
    low  = np.asarray(series_low)
    close = np.asarray(series_close)
    tr1 = high - low
    tr2 = np.abs(high - np.concatenate(([close[0]], close[:-1])))
    tr3 = np.abs(low - np.concatenate(([close[0]], close[:-1])))
    tr = np.vstack([tr1, tr2, tr3]).max(axis=0)
    atr = pd.Series(tr).rolling(period).mean()
    return atr.values

def daily_bias_from_D1(d1_df, now_price):
    """
    ICT-style bias:
    - bias long if price > daily_open and price > prev_day_high or positive recent momentum
    - bias short if price < daily_open and price < prev_day_low or negative momentum
    returns 1 (long), -1 (short), 0 neutral
    """
    today_open = d1_df.iloc[-1]['open']
    prev_high = d1_df.iloc[-2]['high']
    prev_low  = d1_df.iloc[-2]['low']
    c1 = d1_df.iloc[-1]['close']; c2 = d1_df.iloc[-2]['close']; c3 = d1_df.iloc[-3]['close']
    momentum = (c1 - c2) + (c2 - c3)
    if (now_price > today_open and now_price > prev_high) or momentum > 0:
        return 1
    if (now_price < today_open and now_price < prev_low) or momentum < 0:
        return -1
    return 0

def find_swings_levels(df_m15, lookback=120):
    """
    Returns list of swing levels (highs and lows) in last lookback M15 bars
    Very simple approach: local highs/lows
    """
    rs = df_m15.copy().reset_index(drop=True).tail(lookback)
    highs = []
    lows = []
    for i in range(2, len(rs)-2):
        h = rs.loc[rs.index[i], 'high']
        if h > rs.loc[rs.index[i-1], 'high'] and h > rs.loc[rs.index[i+1],'high']:
            highs.append(h)
        l = rs.loc[rs.index[i], 'low']
        if l < rs.loc[rs.index[i-1], 'low'] and l < rs.loc[rs.index[i+1],'low']:
            lows.append(l)
    levels = highs + lows
    levels = sorted(list(set(levels)))
    return levels

def cluster_levels(levels, cluster_pips, point):
    """
    cluster_pips: threshold in pips (points)
    returns zones as list of (low, high)
    """
    if not levels: return []
    levels = sorted(levels)
    zones = []
    cur_low = levels[0]; cur_high = levels[0]
    thresh = cluster_pips * point
    for lv in levels[1:]:
        if abs(lv - cur_high) <= thresh:
            cur_high = max(cur_high, lv)
            cur_low  = min(cur_low, lv)
        else:
            zones.append((cur_low, cur_high))
            cur_low = cur_high = lv
    zones.append((cur_low, cur_high))
    return zones

def is_price_touch_zone(price, zone, buffer_points):
    low, high = zone
    return (price >= (low - buffer_points)) and (price <= (high + buffer_points))

def check_rejection_m1(df_m1_recent, zone, min_wick_pts, point):
    """
    df_m1_recent: pandas DataFrame of most recent M1 candles (newest last)
    We'll check last N candles for long/short rejection depending on bias
    Return True/False
    """
    low, high = zone
    for _, row in df_m1_recent.iterrows():
        body_top = max(row['open'], row['close'])
        body_bot = min(row['open'], row['close'])
        # lower wick
        lower_wick = body_bot - row['low']
        upper_wick = row['high'] - body_top
        if lower_wick/point >= min_wick_pts and row['low'] <= high + point*2 and row['close'] > row['open']:
            return True
        if upper_wick/point >= min_wick_pts and row['high'] >= low - point*2 and row['close'] < row['open']:
            return True
    return False