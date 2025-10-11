import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import warnings

warnings.filterwarnings("ignore")


class ModelTrainer:
    def __init__(self):
        self.models = {}
        self.predictions = {}

    def init_models(self):
        """初始化多个模型"""
        self.models = {
            "ridge": Ridge(alpha=10.0),
            "lasso": Lasso(alpha=0.0005),
            "elasticnet": ElasticNet(alpha=0.001, l1_ratio=0.5),
            "rf": RandomForestRegressor(
                n_estimators=200,
                max_depth=15,
                min_samples_split=5,
                random_state=42,
                n_jobs=-1,
            ),
            "gbr": GradientBoostingRegressor(
                n_estimators=300, learning_rate=0.05, max_depth=4, random_state=42
            ),
            "xgb": XGBRegressor(
                n_estimators=300, learning_rate=0.05, max_depth=4, random_state=42
            ),
            "lgbm": LGBMRegressor(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=4,
                random_state=42,
                verbose=-1,
            ),
        }

        return self.models

    def cross_validate(self, X, y, n_folds=5):
        """交叉验证评估模型"""
        kfold = KFold(n_splits=n_folds, shuffle=True, random_state=42)
        scores = {}

        for name, model in self.models.items():
            cv_scores = cross_val_score(
                model, X, y, scoring="neg_mean_squared_error", cv=kfold, n_jobs=-1
            )
            rmse_scores = np.sqrt(-cv_scores)
            scores[name] = {"mean": rmse_scores.mean(), "std": rmse_scores.std()}
            print(
                f"{name}: RMSE = {rmse_scores.mean():.4f} (+/- {rmse_scores.std():.4f})"
            )

        return scores

    def train_models(self, X_train, y_train):
        """训练所有模型"""
        for name, model in self.models.items():
            print(f"Training {name}...")
            model.fit(X_train, y_train)

        return self.models

    def predict(self, X_test):
        """使用训练好的模型进行预测"""
        for name, model in self.models.items():
            self.predictions[name] = model.predict(X_test)

        return self.predictions

    def ensemble_predict(self, weights=None):
        """集成多个模型的预测结果"""
        if weights is None:
            # 默认权重：更好的模型获得更高权重
            weights = {
                "ridge": 0.05,
                "lasso": 0.05,
                "elasticnet": 0.05,
                "rf": 0.15,
                "gbr": 0.20,
                "xgb": 0.25,
                "lgbm": 0.25,
            }

        ensemble_pred = np.zeros(len(list(self.predictions.values())[0]))

        for name, pred in self.predictions.items():
            ensemble_pred += weights[name] * pred

        return ensemble_pred
