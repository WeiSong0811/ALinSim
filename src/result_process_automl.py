import argparse
import json
# import matplotlib.pyplot as plt
# import numpy as np
import os
import shutil
import numpy as np
import xgboost as xgb
# from sklearn.ensemble import GradientBoostingRegressor
# from autosklearn.regression import AutoSklearnRegressor
from utils import generation_pool_pan, func, predict
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import pandas as pd
from sklearn.preprocessing import StandardScaler
from datetime import datetime
import time
from pathlib import Path
# import tqdm
from sklearn.ensemble import RandomForestRegressor
def custom_rmse(y_true, y_pred, weights=None):
    """
    自定义 RMSE 函数。

    参数:
        y_true (array-like): 真实值。
        y_pred (array-like): 预测值。
        weights (array-like, optional): 权重，如果需要加权 RMSE。

    返回:
        float: 计算得到的 RMSE 值。
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # 检查输入长度是否一致
    if len(y_true) != len(y_pred):
        raise ValueError("y_true 和 y_pred 的长度必须一致！")

    # 计算误差
    errors = y_true - y_pred

    # 加权计算
    if weights is not None:
        weights = np.array(weights)
        if len(weights) != len(y_true):
            raise ValueError("weights 的长度必须和 y_true 一致！")
        mse = np.sum(weights * errors ** 2) / np.sum(weights)
    else:
        mse = np.mean(errors ** 2)

    # 返回 RMSE
    return np.sqrt(mse)

def dataset_generation(idx_list):

    query_X_df = pd.DataFrame(columns=Parameter_space.columns)
    query_y_df = pd.DataFrame(columns=target_variable)
    
    for idx in idx_list:
        X = Parameter_space.loc[idx].values
        y_sim = predict(X)
        # 将查询到的样本添加到查询数据集中,并且要考虑到idx这个绝对索引，不能直接append，要用loc或者iloc来添加
        query_X_df.loc[idx] = Parameter_space.loc[idx]
        query_y_df.loc[idx, target_variable] = y_sim

    return query_X_df, query_y_df

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run AutoSklearn on a specific dataset.")
    parser.add_argument("--random_state", type=int, required=True, help="Random state for fitting.")
    parser.add_argument("--idx", type=int, required=True, help="ID of the strategy to fit.")
    args = parser.parse_args()

    random_state = args.random_state
    data_idx = args.idx

    idx_path = Path(f'../results/result_pan_5/{data_idx}/{random_state}')
    pattern = 'TreeBasedRegressor_Representativity_self_*.json'
    file = sorted(idx_path.glob(pattern))

    with open(file[0], 'r') as f:
        data = json.load(f)

    target_variable = ["wca", "q", "sigma"]
    x_test, X_pool_filtered = generation_pool_pan(seed=random_state)

    y_test = func(x_test.to_numpy(dtype=float))
    y_test = pd.DataFrame(y_test, columns=target_variable)
    y_test = y_test.iloc[:, data_idx]
 
    Parameter_space = X_pool_filtered.copy()

    method_key = next(iter(data.keys()))  # 获取第一个键
    steps = data[method_key]

    idx_list = []
    mae_list = []
    rmse_list = []
    r2_list = []
    for s in steps:
        idx_list += s
        x_train, y_train = dataset_generation(idx_list)
        # print(x_train,y_train)
        y_train = y_train.iloc[:, data_idx]
        # 模型训练和预测
        
        cur_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
        tmp_dir = f"../tmp/autosklearn_{random_state}_{cur_time}_{os.getpid()}"
        if not os.path.exists('../tmp/'):
                os.makedirs('../tmp/')
        # print(X_labeled.shape, y_labeled.shape)
        # model = AutoSklearnRegressor(
        #    time_left_for_this_task=900,  # 总时间限制 2 小时
        #    per_run_time_limit=200,  # 每次拟合限制 5 分钟
        #    n_jobs=-1,  # 使用所有可用线程
        #    seed=random_state,  # 设置随机种子
        #    tmp_folder=tmp_dir,
        #)
        
        model = RandomForestRegressor(n_estimators=200, random_state=random_state)
        
        model.fit(x_train, y_train)
        print('Finisch train', x_train.shape)
        
        print('x_train shape:',x_train.shape, 'y_train shape:', y_train.shape)
        y_pred = model.predict(x_test)

        mae = mean_absolute_error(y_test, y_pred)
        rmse = custom_rmse(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        mae_list.append(mae)
        rmse_list.append(rmse)
        r2_list.append(r2)
    # 保存结果到json

    results = {
        "mae": mae_list,
        "rmse": rmse_list,
        "r2": r2_list
    }
    with open(f'../result_pan_5/AM_AL_results_seed_{random_state}_{data_idx}.json', 'w') as f:
        json.dump(results, f)