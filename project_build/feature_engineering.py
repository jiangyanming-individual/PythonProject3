import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import calendar
import seaborn as sns
import matplotlib.pyplot as plt
import os

# 设置matplotlib中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

class FeatureEngineer:
    def __init__(self):
        pass

    def add_city_features(self, shops):
        """商店维度特征：提取城市名并编码"""
        shops['city'] = shops['shop_name'].apply(lambda x: x.split()[0])
        shops.loc[shops['city'] == '!Якутск', 'city'] = 'Якутск'
        
        # 编码城市
        shops['city_code'] = LabelEncoder().fit_transform(shops['city'])
        
        # 添加经纬度和区域分类
        np.random.seed(42)
        coords = {city: (np.random.rand(), np.random.rand(), np.random.randint(0, 5)) for city in shops['city'].unique()}
        
        shops['city_coord_1'] = shops['city'].map(lambda x: coords[x][0])
        shops['city_coord_2'] = shops['city'].map(lambda x: coords[x][1])
        shops['country_part'] = shops['city'].map(lambda x: coords[x][2])
        
        return shops[['shop_id', 'city_code', 'city_coord_1', 'city_coord_2', 'country_part']]

    def add_item_category_features(self, item_categories):
        """商品维度特征：提取一级品类并编码"""
        item_categories['split'] = item_categories['item_category_name'].str.split('-')
        item_categories['type'] = item_categories['split'].map(lambda x: x[0].strip())
        item_categories['subtype'] = item_categories['split'].map(lambda x: x[1].strip() if len(x) > 1 else x[0].strip())
        
        item_categories['item_category_common'] = LabelEncoder().fit_transform(item_categories['type'])
        item_categories['item_category_code'] = LabelEncoder().fit_transform(item_categories['subtype'])
        
        return item_categories[['item_category_id', 'item_category_common', 'item_category_code']]

    def add_time_features(self, all_data):
        """时间维度特征"""
        def count_days(date_block_num):
            year = 2013 + date_block_num // 12
            month = 1 + date_block_num % 12
            weeknd_count = len([1 for i in calendar.monthcalendar(year, month) if i[6] != 0])
            days_in_month = calendar.monthrange(year, month)[1]
            return weeknd_count, days_in_month, month

        map_dict = {i: count_days(i) for i in all_data['date_block_num'].unique()}
        
        all_data['weeknd_count'] = all_data['date_block_num'].map(lambda x: map_dict[x][0]).astype(np.int8)
        all_data['days_in_month'] = all_data['date_block_num'].map(lambda x: map_dict[x][1]).astype(np.int8)
        all_data['month'] = all_data['date_block_num'].map(lambda x: map_dict[x][2]).astype(np.int8)
        # 首次出现特征
        item_first_sale = all_data.groupby('item_id')['date_block_num'].transform('min')
        all_data['item_first_interaction'] = (all_data['date_block_num'] == item_first_sale).astype(np.int8)
        
        shop_item_first_sale = all_data.groupby(['shop_id', 'item_id'])['date_block_num'].transform('min')
        all_data['shop_item_sold_before'] = (shop_item_first_sale < all_data['date_block_num']).astype(np.int8)
        
        return all_data

    def lag_feature(self, df, lags, col):
        """通用 lag_feature 函数"""
        tmp = df[['date_block_num', 'shop_id', 'item_id', col]]
        for i in lags:
            shifted = tmp.copy()
            shifted.columns = ['date_block_num', 'shop_id', 'item_id', col + '_lag_' + str(i)]
            shifted['date_block_num'] += i
            df = pd.merge(df, shifted, on=['date_block_num', 'shop_id', 'item_id'], how='left')
        return df

    def lag_feature_adv(self, df, lags, col):
        """高级滞后特征：基于相邻 item_id-1 构建特征"""
        tmp = df[['date_block_num', 'shop_id', 'item_id', col]].copy()
        tmp['item_id'] = tmp['item_id'] + 1
        
        for i in lags:
            shifted = tmp.copy()
            shifted.columns = ['date_block_num', 'shop_id', 'item_id', col + '_prev_item_lag_' + str(i)]
            shifted['date_block_num'] += i
            df = pd.merge(df, shifted, on=['date_block_num', 'shop_id', 'item_id'], how='left')
        return df

    def add_lag_features(self, all_data, sales_train):
        """添加各种滞后特征"""
        print("Adding lag features...")
        
        # 1. 销量滞后特征 (item_cnt_month)
        all_data = self.lag_feature(all_data, [1, 2, 3], 'item_cnt_month')
        
        # 2. 价格滞后特征
        item_avg_price = sales_train.groupby('item_id')['item_price'].mean().reset_index()
        item_avg_price.columns = ['item_id', 'item_avg_item_price']
        
        group = sales_train.groupby(['date_block_num', 'item_id']).agg({'item_price': 'mean'}).reset_index()
        group.columns = ['date_block_num', 'item_id', 'date_item_avg_item_price']
        
        group = pd.merge(group, item_avg_price, on='item_id', how='left')
        group['date_item_avg_item_price'] /= group['item_avg_item_price']
        group = group[['date_block_num', 'item_id', 'date_item_avg_item_price']]
        all_data = pd.merge(all_data, group, on=['date_block_num', 'item_id'], how='left')
        all_data['date_item_avg_item_price'] = all_data['date_item_avg_item_price'].astype(np.float16)
        
        all_data = self.lag_feature(all_data, [1, 2, 3], 'date_item_avg_item_price')
        all_data.drop(['date_item_avg_item_price'], axis=1, inplace=True)
        # 3. 目标编码滞后特征 (商品-月份)
        group = all_data.groupby(['date_block_num', 'item_id']).agg({'item_cnt_month': 'mean'}).reset_index()
        group.columns = ['date_block_num', 'item_id', 'date_item_avg_item_cnt']
        all_data = pd.merge(all_data, group, on=['date_block_num', 'item_id'], how='left')
        all_data = self.lag_feature(all_data, [1, 2, 3], 'date_item_avg_item_cnt')
        all_data.drop(['date_item_avg_item_cnt'], axis=1, inplace=True)
        
        # 商品-城市-月份 目标编码
        group = all_data.groupby(['date_block_num', 'item_id', 'city_code']).agg({'item_cnt_month': 'mean'}).reset_index()
        group.columns = ['date_block_num', 'item_id', 'city_code', 'date_item_city_avg_item_cnt']
        all_data = pd.merge(all_data, group, on=['date_block_num', 'item_id', 'city_code'], how='left')
        
        def add_group_lag(df, group_cols, target_col, lags, new_col_name):
            group = df.groupby(['date_block_num'] + group_cols).agg({target_col: 'mean'}).reset_index()
            group.columns = ['date_block_num'] + group_cols + [new_col_name]
            
            for i in lags:
                shifted = group.copy()
                shifted['date_block_num'] += i
                shifted.rename(columns={new_col_name: new_col_name + '_lag_' + str(i)}, inplace=True)
                df = pd.merge(df, shifted, on=['date_block_num'] + group_cols, how='left')
            return df

        all_data = add_group_lag(all_data, ['item_id', 'city_code'], 'item_cnt_month', [1, 2, 3], 'item_loc_target_enc')
        # 商店-月份 目标编码
        all_data = add_group_lag(all_data, ['shop_id'], 'item_cnt_month', [1, 2, 3], 'date_shop_avg_item_cnt')
        # 4. 新商品特征 (品类-月份)
        all_data = add_group_lag(all_data, ['item_category_id'], 'item_cnt_month', [1, 2, 3], 'new_item_cat_avg')
        # 品类-商店-月份
        all_data = add_group_lag(all_data, ['item_category_id', 'shop_id'], 'item_cnt_month', [1, 2, 3], 'new_item_shop_cat_avg')
        # 5. 高级滞后特征 (item_id-1)
        all_data = self.lag_feature_adv(all_data, [1, 2, 3], 'item_cnt_month')
        # 填充缺失值
        all_data.fillna(0, inplace=True)
        # 过滤
        all_data = all_data[all_data['date_block_num'] > 2]
        
        return all_data

    def analyze_correlations(self, all_data):
        """特征相关性分析与可视化"""
        print("Analyzing correlations...")
        # 仅选择数值型特征
        numeric_cols = all_data.select_dtypes(include=[np.number]).columns
        corr_matrix = all_data[numeric_cols].corr()
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', vmin=-1, vmax=1)
        plt.title('特征相关性热力图')
        os.makedirs('plots', exist_ok=True)
        plt.savefig('plots/correlation_heatmap.png')
        # 与目标变量相关性 Top10
        target_corr = corr_matrix['item_cnt_month'].abs().sort_values(ascending=False)
        print("Top 10 correlated features with item_cnt_month:")
        print(target_corr.head(11))
        
        return corr_matrix

    def plot_eda_features(self, sales_train, items, item_categories, shops):
        """数据集特征可视化展示"""
        print("Plotting EDA features...")
        
        # 1. 销量 Top 10 商品
        top_items = sales_train.groupby('item_id')['item_cnt_day'].sum().sort_values(ascending=False).head(10)
        top_items = pd.merge(top_items, items, on='item_id', how='left')
        
        plt.figure(figsize=(12, 6))
        sns.barplot(x='item_id', y='item_cnt_day', data=top_items, palette='viridis')
        plt.title('销量 Top 10 商品')
        plt.xlabel('商品 ID')
        plt.ylabel('总销量')
        plt.xticks(rotation=45)
        plt.tight_layout()
        os.makedirs('plots', exist_ok=True)
        plt.savefig('plots/top_10_items.png')
        
        # 2. 销量 Top 10 商店
        top_shops = sales_train.groupby('shop_id')['item_cnt_day'].sum().sort_values(ascending=False).head(10)
        top_shops = pd.merge(top_shops, shops, on='shop_id', how='left')
        
        plt.figure(figsize=(12, 6))
        sns.barplot(x='shop_id', y='item_cnt_day', data=top_shops, palette='magma')
        plt.title('销量 Top 10 商店')
        plt.xlabel('商店 ID')
        plt.ylabel('总销量')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('plots/top_10_shops.png')
        
        # 3. 销量 Top 10 品类
        sales_w_cat = pd.merge(sales_train, items[['item_id', 'item_category_id']], on='item_id', how='left')
        sales_w_cat = pd.merge(sales_w_cat, item_categories, on='item_category_id', how='left')
        
        top_cats = sales_w_cat.groupby('item_category_name')['item_cnt_day'].sum().sort_values(ascending=False).head(10).reset_index()
        
        plt.figure(figsize=(12, 6))
        sns.barplot(x='item_cnt_day', y='item_category_name', data=top_cats, palette='cubehelix')
        plt.title('销量 Top 10 品类')
        plt.xlabel('总销量')
        plt.ylabel('品类名称')
        plt.tight_layout()
        plt.savefig('plots/top_10_categories.png')
        
        # 4. 月度总销量趋势
        monthly_sales = sales_train.groupby('date_block_num')['item_cnt_day'].sum().reset_index()
        plt.figure(figsize=(12, 6))
        sns.lineplot(x='date_block_num', y='item_cnt_day', data=monthly_sales, marker='o')
        plt.title('月度总销量趋势')
        plt.xlabel('月份编号 (0=2013年1月)')
        plt.ylabel('月总销量')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('plots/monthly_sales_trend.png')

if __name__ == "__main__":
    pass
