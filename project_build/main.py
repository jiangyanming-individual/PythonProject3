import pandas as pd
import numpy as np
import gc
import os
import matplotlib.pyplot as plt
import seaborn as sns
from data_loader import DataLoader
from feature_engineering import FeatureEngineer
import xgboost as xgb
import lightgbm as lgb
from model import ModelTrainer

# 设置路径
DATA_PATH = "../predict_future_sales_data"
DEBUG = False  # 设置为TRUE 进行调试

def main():

    # 1. 数据加载与预处理
    loader = DataLoader(DATA_PATH)
    sales_train, test, items, item_categories, shops = loader.load_data(debug=DEBUG)
    sales_train = loader.preprocess_data(sales_train, test)
    
    # 2. 数据重构
    all_data = loader.restructure_data(sales_train, test)
    # 3. 特征工程
    fe = FeatureEngineer()
    # EDA 可视化展示
    fe.plot_eda_features(sales_train, items, item_categories, shops)
    # 商店特征
    shops_features = fe.add_city_features(shops)
    all_data = pd.merge(all_data, shops_features, on='shop_id', how='left')
    # 商品特征
    item_cat_features = fe.add_item_category_features(item_categories)
    items = pd.merge(items, item_cat_features, on='item_category_id', how='left')
    all_data = pd.merge(all_data, items[['item_id', 'item_category_id', 'item_category_common', 'item_category_code']], on='item_id', how='left')
    # 时间特征
    all_data = fe.add_time_features(all_data)
    # 滞后特征
    all_data = fe.add_lag_features(all_data, sales_train)
    # 释放内存
    del sales_train, shops, items, item_categories, shops_features, item_cat_features
    gc.collect()
    # 特征相关性分析
    # fe.analyze_correlations(all_data)
    # 4. 数据集划分
    max_train_month = all_data.loc[all_data['date_block_num'] < 34, 'date_block_num'].max()
    valid_month = max_train_month
    print(f"Using month {valid_month} as validation set (derived from data).")

    X_train = all_data[all_data.date_block_num < valid_month].drop(['item_cnt_month'], axis=1)
    y_train = all_data[all_data.date_block_num < valid_month]['item_cnt_month']
    
    X_valid = all_data[all_data.date_block_num == valid_month].drop(['item_cnt_month'], axis=1)
    y_valid = all_data[all_data.date_block_num == valid_month]['item_cnt_month']
    
    X_test = all_data[all_data.date_block_num == 34].drop(['item_cnt_month'], axis=1)

    print(f"X_train shape: {X_train.shape}")
    print(f"X_valid shape: {X_valid.shape}")
    print(f"X_test shape: {X_test.shape}")
    
    if X_train.empty or X_valid.empty:
        raise ValueError("Training or Validation set is empty! Check data range and date_block_num.")
    test_id = test['ID'] if 'ID' in test.columns else test.index
    
    # 释放 all_data
    del all_data
    gc.collect()
    
    # 特征相关性分析
    fe.analyze_correlations(pd.concat([X_train, y_train], axis=1))
    # 5. 模型构建与评估
    trainer = ModelTrainer()
    # 单模型 CV 评估
    print("\n=== Cross Validation ===")
    cv_scores_df, cv_models = trainer.time_series_cv(pd.concat([X_train, X_valid]), pd.concat([y_train, y_valid]), model_type='lgb')
    # 保存 CV 指标
    cv_scores_df.to_csv('cv_results.csv', index=False)
    print("CV results saved to cv_results.csv")
    # 单模型训练与验证
    print("\n=== Single Model Training ===")
    model_metrics = []  # 收集所有模型指标
    # LightGBM
    lgb_model = trainer.train_lightgbm(X_train, y_train, X_valid, y_valid)
    lgb_pred = lgb_model.predict(X_valid)
    lgb_metrics = trainer.get_metrics(y_valid, lgb_pred)
    lgb_metrics['Model'] = 'LightGBM'
    model_metrics.append(lgb_metrics)
    print("LightGBM Valid Metrics:", lgb_metrics)
    trainer.plot_feature_importance(lgb_model, X_train.columns, 'LightGBM')
    trainer.plot_residuals(y_valid, lgb_pred, 'LightGBM')
    # XGBoost
    xgb_model = trainer.train_xgboost(X_train, y_train, X_valid, y_valid)
    xgb_pred = xgb_model.predict(xgb.DMatrix(X_valid))
    xgb_metrics = trainer.get_metrics(y_valid, xgb_pred)
    xgb_metrics['Model'] = 'XGBoost'
    model_metrics.append(xgb_metrics)
    print("XGBoost Valid Metrics:", xgb_metrics)
    trainer.plot_feature_importance(xgb_model, X_train.columns, 'XGBoost')
    # Random Forest
    rf_model = trainer.train_rf(X_train.fillna(-999), y_train)
    rf_pred = rf_model.predict(X_valid.fillna(-999))
    rf_metrics = trainer.get_metrics(y_valid, rf_pred)
    rf_metrics['Model'] = 'RandomForest'
    model_metrics.append(rf_metrics)
    print("Random Forest Valid Metrics:", rf_metrics)
    trainer.plot_feature_importance(rf_model, X_train.columns, 'RandomForest')
    # Stacking
    print("\n=== Stacking Model ===")
    meta_model, final_pred, models = trainer.stacking_train(X_train.fillna(-999), y_train, X_valid.fillna(-999), y_valid, X_test.fillna(-999))
    # Stacking 评估
    lgb_val = lgb_model.predict(X_valid)
    xgb_val = xgb_model.predict(xgb.DMatrix(X_valid))
    rf_val = rf_model.predict(X_valid.fillna(-999))
    stack_val = np.column_stack((lgb_val, xgb_val, rf_val))
    stack_pred_val = meta_model.predict(stack_val)
    
    stack_metrics = trainer.get_metrics(y_valid, stack_pred_val)
    stack_metrics['Model'] = 'Stacking'
    model_metrics.append(stack_metrics)
    print("Stacking Model Valid Metrics:", stack_metrics)
    trainer.plot_residuals(y_valid, stack_pred_val, 'Stacking')
    
    # 保存所有模型指标到 CSV
    metrics_df = pd.DataFrame(model_metrics)
    metrics_df = metrics_df[['Model', 'RMSE', 'MAE', 'R2']]
    metrics_df.to_csv('model_metrics.csv', index=False)
    print("Model metrics saved to model_metrics.csv")
    # 提交文件
    submission = pd.DataFrame({
        "ID": test_id,
        "item_cnt_month": final_pred.clip(0, 20)
    })
    submission.to_csv('submission.csv', index=False)
    print("Submission file saved to submission.csv")
    
    # 绘制指标对比图
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Model', y='RMSE', data=metrics_df)
    plt.title('模型 RMSE 对比')
    plt.ylabel('均方根误差 (RMSE)')
    os.makedirs('plots', exist_ok=True)
    plt.savefig('plots/model_comparison.png')

if __name__ == "__main__":
    main()
