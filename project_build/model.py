import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import xgboost as xgb
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

class ModelTrainer:
    def __init__(self):
        pass

    def get_metrics(self, y_true, y_pred):
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        return {'RMSE': rmse, 'R2': r2, 'MAE': mae}

    def train_lightgbm(self, X_train, y_train, X_valid, y_valid):
        print("Training LightGBM...")
        lgb_params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'learning_rate': 0.05,
            'num_leaves': 31,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'seed': 42,
            'n_jobs': -1,
            'verbose': -1
        }
        
        dtrain = lgb.Dataset(X_train, label=y_train)
        dvalid = lgb.Dataset(X_valid, label=y_valid, reference=dtrain)
        
        # 参数调整
        model = lgb.train(
            lgb_params,
            dtrain,
            num_boost_round=1000,
            valid_sets=[dtrain, dvalid],
            callbacks=[lgb.early_stopping(stopping_rounds=20), lgb.log_evaluation(period=50)]
        )
        return model

    def train_xgboost(self, X_train, y_train, X_valid, y_valid):
        print("Training XGBoost...")
        xgb_params = {
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'learning_rate': 0.05,
            'max_depth': 8,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'seed': 42,
            'n_jobs': -1
        }
        
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dvalid = xgb.DMatrix(X_valid, label=y_valid)
        
        model = xgb.train(
            xgb_params,
            dtrain,
            num_boost_round=1000,
            evals=[(dtrain, 'train'), (dvalid, 'valid')],
            early_stopping_rounds=20,
            verbose_eval=50
        )
        return model

    def train_rf(self, X_train, y_train):
        print("Training Random Forest...")
        # 限制深度和树数量以加快速度
        rf = RandomForestRegressor(n_estimators=50, max_depth=10, n_jobs=-1, random_state=42)
        rf.fit(X_train, y_train)
        return rf

    def time_series_cv(self, X, y, model_type='lgb'):
        tscv = TimeSeriesSplit(n_splits=5)
        scores = []
        models = []
        
        for fold, (train_index, val_index) in enumerate(tscv.split(X)):
            print(f"Fold {fold+1}")
            X_train_fold, X_val_fold = X.iloc[train_index], X.iloc[val_index]
            y_train_fold, y_val_fold = y.iloc[train_index], y.iloc[val_index]
            
            if model_type == 'lgb':
                model = self.train_lightgbm(X_train_fold, y_train_fold, X_val_fold, y_val_fold)
                pred = model.predict(X_val_fold)
            elif model_type == 'xgb':
                model = self.train_xgboost(X_train_fold, y_train_fold, X_val_fold, y_val_fold)
                pred = model.predict(xgb.DMatrix(X_val_fold))
            elif model_type == 'rf':
                model = self.train_rf(X_train_fold, y_train_fold)
                pred = model.predict(X_val_fold)
            
            metrics = self.get_metrics(y_val_fold, pred)
            print(f"Metrics: {metrics}")
            metrics['Fold'] = fold + 1
            scores.append(metrics)
            models.append(model)
            
        return pd.DataFrame(scores), models

    def stacking_train(self, X_train, y_train, X_valid, y_valid, X_test):
        print("Training Stacking Model...")
        
        # 第一层模型训练
        # 1. LightGBM
        lgb_model = self.train_lightgbm(X_train, y_train, X_valid, y_valid)
        lgb_pred_val = lgb_model.predict(X_valid)
        lgb_pred_test = lgb_model.predict(X_test)
        
        # 2. XGBoost
        xgb_model = self.train_xgboost(X_train, y_train, X_valid, y_valid)
        xgb_pred_val = xgb_model.predict(xgb.DMatrix(X_valid))
        xgb_pred_test = xgb_model.predict(xgb.DMatrix(X_test))
        
        # 3. Random Forest
        rf_model = self.train_rf(X_train, y_train)
        rf_pred_val = rf_model.predict(X_valid)
        rf_pred_test = rf_model.predict(X_test)
        
        # 第二层特征
        stack_X_val = np.column_stack((lgb_pred_val, xgb_pred_val, rf_pred_val))
        stack_X_test = np.column_stack((lgb_pred_test, xgb_pred_test, rf_pred_test))
        
        # 第二层模型：线性回归
        meta_model = LinearRegression()
        meta_model.fit(stack_X_val, y_valid)
        
        final_pred = meta_model.predict(stack_X_test)
        
        return meta_model, final_pred, (lgb_model, xgb_model, rf_model)

    def plot_feature_importance(self, model, features, model_name):
        plt.figure(figsize=(10, 12))
        if model_name == 'LightGBM':
            lgb.plot_importance(model, max_num_features=30, figsize=(10, 12))
            plt.title('LightGBM 特征重要性')
            plt.xlabel('特征重要性')
            plt.ylabel('特征')
        elif model_name == 'XGBoost':
            xgb.plot_importance(model, max_num_features=30)
            plt.title('XGBoost 特征重要性')
            plt.xlabel('F得分')
            plt.ylabel('特征')
        elif model_name == 'RandomForest':
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1][:30]
            plt.title('随机森林 特征重要性')
            plt.barh(range(len(indices)), importances[indices], align='center')
            plt.yticks(range(len(indices)), [features[i] for i in indices])
            plt.xlabel('相对重要性')
        
        plt.tight_layout()
        os.makedirs('plots', exist_ok=True)
        plt.savefig(f'plots/{model_name}_importance.png')

    def plot_residuals(self, y_true, y_pred, model_name):
        residuals = y_true - y_pred
        plt.figure(figsize=(10, 6))
        sns.histplot(residuals, kde=True)
        plt.title(f'{model_name} 残差分布')
        plt.xlabel('残差')
        plt.ylabel('频率')
        os.makedirs('plots', exist_ok=True)
        plt.savefig(f'plots/{model_name}_residuals.png')

if __name__ == "__main__":
    pass
