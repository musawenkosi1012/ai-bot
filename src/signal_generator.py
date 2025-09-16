# src/signal_generator.py
import numpy as np
from indicators import atr, find_swings_levels, cluster_levels, is_price_touch_zone, check_rejection_m1

def generate_candidate(df_m1, df_m15, df_d1, point, ml_inference_func, params):
    """
    params: dict with SR clustering settings, ATR multipliers, buffer, rejection params, thresholds
    ml_inference_func: function(features)->dict {'p_win':..., 'pred_slippage':...}
    Returns: dict with trade decision or None
    """
    price = df_m1.iloc[-1]['close']
    bias = params['daily_bias']
    # build zones
    levels = find_swings_levels(df_m15, lookback=params['sr_lookback'])
    zones = cluster_levels(levels, params['sr_cluster_pips'], point)
    if not zones: return None

    # pick zone based on bias
    target_zone = None
    if bias == 1:
        # pick nearest zone below price
        below = [z for z in zones if (z[0]+z[1])/2.0 < price]
        if not below: return None
        target_zone = sorted(below, key=lambda z: price - (z[0]+z[1])/2.0)[0]
    elif bias == -1:
        above = [z for z in zones if (z[0]+z[1])/2.0 > price]
        if not above: return None
        target_zone = sorted(above, key=lambda z: (z[0]+z[1])/2.0 - price)[0]
    else:
        # neutral: ignore if strict mode
        if params.get('use_daily_bias_only', True):
            return None
        target_zone = sorted(zones, key=lambda z: abs(price - ((z[0]+z[1])/2.0)))[0]

    # check touch
    if not is_price_touch_zone(price, target_zone, params['zone_buffer_points'] * point):
        return None

    # rejection check
    if params['require_rejection']:
        df_recent = df_m1.tail(params['rejection_candles'])
        if not check_rejection_m1(df_recent, target_zone, params['rejection_wick_pts'], point):
            return None

    # compute features for ML
    features = {}
    # fill features like ATR, distance_to_zone, spread etc. You'll compute them here:
    features['atr_m1'] = atr(df_m1['high'], df_m1['low'], df_m1['close'], period=params['atr_period'])[-1]
    mid = (price + price) / 2.0
    zone_mid = (target_zone[0] + target_zone[1]) / 2.0
    features['dist_zone_pts'] = abs(mid - zone_mid) / point
    features['zone_width_pts'] = (target_zone[1] - target_zone[0]) / point
    features['planned_rr'] = params['tp_mult']/params['sl_mult']
    # Add more features for ML
    features['spread_pts'] = params.get('spread_pts', 1.0)  # Default spread
    features['hour_of_day'] = df_m1.iloc[-1]['timestamp'].hour if 'timestamp' in df_m1.columns else 12

    # call ML inference to score this candidate
    ml_out = ml_inference_func(features)  # expected {'p_win':..., 'pred_slippage':...}
    if ml_out['p_win'] >= params['p_threshold'] and ml_out['pred_slippage'] <= params['max_pred_slippage_pts']:
        # accept
        return {
            'side': 'buy' if bias==1 else 'sell',
            'entry_price': price,
            'zone': target_zone,
            'features': features,
            'ml': ml_out
        }
    return None