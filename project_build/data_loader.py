import pandas as pd
import numpy as np
import os
from itertools import product

class DataLoader:
    def __init__(self, data_path):
        self.data_path = data_path

    def load_data(self, debug=False):
        """加载原始数据"""

        nrows = 10000 if debug else None
        sales_train = pd.read_csv(os.path.join(self.data_path, 'sales_train.csv'), nrows=nrows)
        test = pd.read_csv(os.path.join(self.data_path, 'test.csv'), nrows=nrows)
        items = pd.read_csv(os.path.join(self.data_path, 'items.csv'))
        item_categories = pd.read_csv(os.path.join(self.data_path, 'item_categories.csv'))
        shops = pd.read_csv(os.path.join(self.data_path, 'shops.csv'))
        return sales_train, test, items, item_categories, shops

    def preprocess_data(self, sales_train, test):
        """数据清洗与异常值处理"""
        print("Preprocessing data...")
        # 移除异常值
        sales_train = sales_train[sales_train.item_price < 100000]
        sales_train = sales_train[sales_train.item_price > 0]
        sales_train = sales_train[sales_train.item_cnt_day < 1001]
        
        return sales_train

    def restructure_data(self, sales_train, test):
        """数据重构：构建全量时间序列样本"""
        print("Restructuring data...")
        index_cols = ['shop_id', 'item_id', 'date_block_num']

        # 构建全量组合
        grid = []
        for block_num in sales_train['date_block_num'].unique():
            cur_shops = sales_train.loc[sales_train['date_block_num'] == block_num, 'shop_id'].unique()
            cur_items = sales_train.loc[sales_train['date_block_num'] == block_num, 'item_id'].unique()
            grid.append(np.array(list(product(*[cur_shops, cur_items, [block_num]])), dtype='int16'))

        grid = pd.DataFrame(np.vstack(grid), columns=index_cols, dtype=np.int16)

        # 计算月度销量
        group = sales_train.groupby(index_cols).agg({'item_cnt_day': 'sum'})
        group.columns = ['item_cnt_month']
        group.reset_index(inplace=True)

        # 合并全量组合与月度销量
        all_data = pd.merge(grid, group, on=index_cols, how='left')
        all_data['item_cnt_month'] = (all_data['item_cnt_month']
                                      .fillna(0)
                                      .clip(0, 20)
                                      .astype(np.float16))

        # 处理测试集
        test['date_block_num'] = 34
        test['date_block_num'] = test['date_block_num'].astype(np.int8)
        test['shop_id'] = test['shop_id'].astype(np.int8)
        test['item_id'] = test['item_id'].astype(np.int16)
        
        # 合并训练集和测试集
        all_data = pd.concat([all_data, test], ignore_index=True, sort=False, keys=index_cols)
        all_data.fillna(0, inplace=True)
        
        return all_data

if __name__ == "__main__":

    # 测试代码
    data_path = "../predict_future_sales_data"
    loader = DataLoader(data_path)
    sales_train, test, items, item_categories, shops = loader.load_data()
    sales_train = loader.preprocess_data(sales_train, test)
    all_data = loader.restructure_data(sales_train, test)
    print(all_data.head())
    print(all_data.info())
