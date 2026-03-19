import argparse
import json
# import matplotlib.pyplot as plt
# import numpy as np
import os
import numpy as np
# from sklearn.ensemble import GradientBoostingRegressor
from autosklearn.regression import AutoSklearnRegressor
from sklearn.metrics import r2_score, mean_absolute_error
import pandas as pd
from sklearn.preprocessing import StandardScaler
import time
from pathlib import Path
# from sklearn.ensemble import RandomForestRegressor

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


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Run AutoSklearn on a specific dataset.")
    parser.add_argument("--random_state", type=int, required=True, help="Random state for fitting.")
    parser.add_argument("--round_idx", type=int, required=True, help="ID of the strategy to fit.")
    args = parser.parse_args()

    random_state = args.random_state
    idx_path = Path(f'../results/result_single_trc/{random_state}')
    round_idx = args.round_idx
    target_variable = ['f_res']

    fea_df = pd.read_csv(f'./run_trc/predicted_datasets/sim_trc_{random_state}.csv')
    x_test = fea_df.drop(columns=target_variable)
    y_test = fea_df[target_variable]

    train_path = Path(f'../results/result_single_trc/{random_state}/xy_record')
    x_train = pd.read_csv(os.path.join(train_path,f'round_{round_idx:02d}_X.csv'), index_col=0)
    y_train = pd.read_csv(os.path.join(train_path,f'round_{round_idx:02d}_y.csv'), index_col=0)
    print(x_train)
    if x_train.shape[0] == (round_idx + 1) * 5:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(x_train)
        x_test_scaled = scaler.transform(x_test)
        cur_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
        tmp_dir = f"../tmp/autosklearn_{random_state}_{cur_time}_{os.getpid()}"
        if not os.path.exists('../tmp/'):
                os.makedirs('../tmp/')
        # print(X_labeled.shape, y_labeled.shape)
        model = AutoSklearnRegressor(
            time_left_for_this_task=900,  # 总时间限制 2 小时
            per_run_time_limit=200,  # 每次拟合限制 5 分钟
            n_jobs=-1,  # 使用所有可用线程
            seed=random_state,  # 设置随机种子
            tmp_folder=tmp_dir,
        )
        
        # model = RandomForestRegressor(n_estimators=200, random_state=random_state)
        
        model.fit(X_train_scaled, y_train)
        print('Finisch train', X_train_scaled.shape)
        
        print('x_train shape:',X_train_scaled.shape, 'y_train shape:', y_train.shape)
        y_pred = model.predict(x_test_scaled)

        mae = mean_absolute_error(y_test, y_pred)
        rmse = custom_rmse(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        results = {
            "mae": mae,
            "rmse": rmse,
            "r2": r2
        }
        with open(f'../result_pan_5/AM_AL_results_seed_{random_state}_{round_idx:02d}.json', 'w') as f:
            json.dump(results, f)
    else:
        print(f'[ERROR]: The number of training parameters in the {round_idx} round is incorrect.')
        