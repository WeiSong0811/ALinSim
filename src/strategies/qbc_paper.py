import numpy as np
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor


class QBC_Paper:
    """
    Paper-style QBC for single-output regression only
    Committee = {RF, XGB}
    Query score = |pred_rf - pred_xgb|
    """

    def __init__(self, random_state=None, rf_params=None, xgb_params=None):
        self.random_state = random_state

        # RF settings (paper-style approximation)
        self.rf_params = rf_params or {
            "n_estimators": 100,
            "max_features": 0.3,   # 30% features at each split
            "random_state": self.random_state,
            "n_jobs": -1
        }

        # XGB settings (from paper description)
        self.xgb_params = xgb_params or {
            "booster": "gbtree",
            "n_estimators": 1000,
            "learning_rate": 0.1,
            "reg_alpha": 0.01,      # L1
            "reg_lambda": 0.1,      # L2
            "tree_method": "hist",
            "subsample": 0.85,
            "colsample_bytree": 0.3,
            "colsample_bylevel": 0.5,
            "random_state": self.random_state,
            "n_jobs": -1
        }

        self.rf = RandomForestRegressor(**self.rf_params)
        self.xgb = XGBRegressor(**self.xgb_params)

    def fit(self, X, y):
        # 单输出：直接训练（假设 y 已经是一维）
        self.rf.fit(X, y)
        self.xgb.fit(X, y)

    def disagreement_scores(self, X_unlabeled):
        pred_rf = self.rf.predict(X_unlabeled)    # shape: (n_samples,)
        pred_xgb = self.xgb.predict(X_unlabeled)  # shape: (n_samples,)
        scores = np.abs(pred_rf - pred_xgb)
        return scores

    def query(self, X_unlabeled, n_act, X_labeled, y_labeled, y_unlabeled=None):
        """
        Return absolute indices from X_unlabeled.index
        """
        # 1) fit committee on current labeled set
        self.fit(X_labeled, y_labeled)

        # 2) compute disagreement
        scores = self.disagreement_scores(X_unlabeled)

        # 3) top-k by disagreement (descending)
        n_act = min(n_act, len(X_unlabeled))
        query_idx_local = np.argsort(scores)[-n_act:][::-1] 

        # 4) return absolute indices (DataFrame index)
        selected_indices = [int(i) for i in X_unlabeled.index[query_idx_local]]

        return selected_indices