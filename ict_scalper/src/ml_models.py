# src/ml_models.py
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split

def train_models(features_csv_path, out_dir):
    df = pd.read_csv(features_csv_path, parse_dates=['timestamp'])
    # Preprocess - choose features and label columns
    available_cols = ['atr_m1','distance_to_nearest_zone_pts','zone_width_pts','planned_rr','spread_pts','hour_of_day']
    X = df[available_cols].fillna(0)
    y = df['win'].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)
    clf = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42)
    clf.fit(X_train, y_train)
    joblib.dump(clf, f"{out_dir}/clf_win.joblib")
    # slippage regression
    from sklearn.linear_model import Ridge
    X2 = X; y2 = df['slippage_pts'].fillna(0)
    X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, shuffle=False, test_size=0.2)
    reg = RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42)
    reg.fit(X2_train, y2_train)
    joblib.dump(reg, f"{out_dir}/reg_slip.joblib")
    print("Training complete. Models saved to", out_dir)

class MLInference:
    def __init__(self, clf_path, reg_path):
        self.clf = joblib.load(clf_path)
        self.reg = joblib.load(reg_path)

    def predict(self, features_dict):
        # map features to vector in same order used for training
        X = np.array([[features_dict.get('atr_m1',0),
                       features_dict.get('distance_to_nearest_zone_pts',0),
                       features_dict.get('zone_width_pts',0),
                       features_dict.get('planned_rr',1),
                       features_dict.get('spread_pts',0),
                       features_dict.get('hour_of_day',0)]])
        p_win = self.clf.predict_proba(X)[0,1]
        pred_slip = self.reg.predict(X)[0]
        return {'p_win': float(p_win), 'pred_slippage': float(pred_slip)}