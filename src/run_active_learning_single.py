import random
import argparse
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.gaussian_process.kernels import RBF, Matern, ConstantKernel as C, WhiteKernel
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import json
from utils import data_process, data_process_meta, active_learning_auto_single , generation_pool, func
from strategies import TreeBasedRegressor_Representativity_self
import warnings
# from config_loader import load_config, create_kernel
# from sklearn.ensemble import RandomForestRegressor
import pickle
from datetime import datetime
import os
 
def main():  
    # 忽略所有警告
    warnings.filterwarnings("ignore")
    # # 加载配置文件
    # config = load_config()
    
    # initial_method = "random"  # greedy_search kmeans ncc random
    # test_method = ["normal",
                   # "LOO_k-Fold-CV",
    #               "k-Fold-CV",
    #               "RSValidation"]
    
    parser = argparse.ArgumentParser(description="Run AutoSklearn on a specific dataset.")
    parser.add_argument("--random_state", type=int, required=True, help="Random state for fitting.")
    parser.add_argument("--label_idx", type=int, default=0, help="Index of the target variable to fit.")
    # parser.add_argument("--initial-method", type=str, default='random', help="Initial method for active learning.")
    # parser.add_argument("--strategy-idx", type=int, required=True, help="ID of the strategy to fit.")
    # parser.add_argument("--dataset-idx", type=int, required=True, help="ID of the dataset to fit.")
    # parser.add_argument("--n_pro_query", type=int, default=10, help="Number of queries per iteration.")
    # parser.add_argument("--time-limit", type=int, default=3600, help="Time limit for fitting in seconds.")
    args = parser.parse_args()
    random_state = args.random_state
    # initial_method = args.initial_method
    # strategy_num_int = args.strategy_idx
    strategy_num_int = 0
    label_idx = args.label_idx
    # datasets_num_int = args.dataset_idx
    # n_pro_query = args.n_pro_query
    n_pro_query = 2
    # # kernel = create_kernel(config)
    # kernel = RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e3)) + WhiteKernel(noise_level=1,
    #                                                                               noise_level_bounds=(1e-10, 1e+1))
    # kernel = (C(1.0, (1e-2, 1e2)) * RBF(1.0, (1e-2, 1e2)) +
    #           WhiteKernel(noise_level=1, noise_level_bounds=(1e-10, 1e+1)))
    threshold = 0.85 
    # 打印验证加载的参数
    print(f"Random State: {random_state}")
    # print(f"Initial Method: {initial_method}")
    # print(f"Kernel: {kernel}")
    # print(f"Threshold: {threshold}")
    
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    # 设置随机种子
    def set_seed(seed):
        np.random.seed(seed)
        random.seed(seed)
    
    set_seed(random_state)
    # 定义主动学习策略
    estimators = [TreeBasedRegressor_Representativity_self(random_state=random_state, min_samples_leaf=5)]
    
    # if not os.path.exists(f"../result/random_state_{random_state}_{initial_method}"):
    #     os.makedirs(f"../result/random_state_{random_state}_{initial_method}")
    
    X_test, X_pool_filtered = generation_pool(seed=random_state)
    
    # 创建输出文件夹
    result_path = f"../result_review/"
    result_path = os.path.join(result_path, str(label_idx), str(random_state))
    
    result_time_record_path = os.path.join(result_path, 'time_record')
    if not os.path.exists(result_time_record_path):
        os.makedirs(result_time_record_path, exist_ok=True)
    
    if not os.path.exists(result_path):
        os.makedirs(result_path, exist_ok=True)
        
    print(f'result_path: {result_path}')
    print(f'result_time_record_path: {result_time_record_path}')
    
    results = {}
    
    # datasets_list = list(X_pool_filtered.items())
    # key = datasets_list[0]
    # value = datasets_list[1]
    
    estimators_list = [estimators[strategy_num_int]]
    
    # print('*' * 15, f'Processing {key} dataset')
    
    X_t = X_pool_filtered
    X_val = X_test.copy()
    y_val = func(X_val.to_numpy(dtype=float))
    y_val = pd.DataFrame(y_val, columns=["wca", "q", "sigma"])
    # y_val_out = pd.concat([X_test.reset_index(drop=True), y_val], axis=1)
    # y_val_out.to_csv(f'../data/pan_test.csv', index=False)
    # X_t, X_val, y_t, y_val = value
    
    duplicate_samples = X_t.duplicated()
    # 检查X_t的数据量
    total_data_volume = X_t.shape[0]
    n_initial = 5
    # n_pro_query = 10
    # n_queries = (total_data_volume - n_initial - 1) // n_pro_query
    n_queries = 20
    print(f"Total data volume: {total_data_volume}, n_initial: {n_initial}, n_pro_query: {n_pro_query}, "
          f"n_queries: {n_queries}")
    
    result_filename = f"/{estimators[strategy_num_int].__class__.__name__}"
    print(f"Result filename: {result_filename}")
    
    query_idx_all, query_time_all, metrics_all = active_learning_auto_single(estimators_list, X_t, X_val, y_val.iloc[:, label_idx], n_initial,
                                                                       n_pro_query, n_queries, threshold, label_idx=label_idx,
                                                                       random_state=random_state)
    
    result = query_idx_all.copy()
    result['random_state'] = random_state
    
    result_time_record = query_time_all.copy()
    result_time_record['random_state'] = random_state
    
    # 保存结果到json文件
    
    result_json_filename = os.path.join(result_path, f"{estimators[strategy_num_int].__class__.__name__}_{current_time}.json")
    with open(result_json_filename, 'w', encoding='utf-8') as f:
        json.dump(result, f)
        print(f"Dictionary saved to {result_json_filename}")
    
    result_time_record_json_filename = os.path.join(result_time_record_path, f"{estimators[strategy_num_int].__class__.__name__}_{current_time}.json")
    with open(result_time_record_json_filename, 'w', encoding='utf-8') as f:
        json.dump(result_time_record, f)
        print(f"Dictionary saved to {result_time_record_json_filename}")
    
    metrics_json_filename = os.path.join(result_path, f"{estimators[strategy_num_int].__class__.__name__}_metrics_{current_time}.json")
    with open(metrics_json_filename, 'w', encoding='utf-8') as f:
        json.dump(metrics_all, f)
        print(f"Dictionary saved to {metrics_json_filename}")
    
    # Plot metrics curves by iteration: R2, MAE, RMSE
    strategy_name = next(iter(metrics_all))
    metrics_seq = metrics_all[strategy_name]
    iterations = list(range(len(metrics_seq)))
    r2_values = [m["r2"] for m in metrics_seq]
    mae_values = [m["mae"] for m in metrics_seq]
    rmse_values = [m["rmse"] for m in metrics_seq]
    
    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    
    axes[0].plot(iterations, r2_values, marker='o')
    axes[0].set_title('R2 vs Iteration')
    axes[0].set_ylabel('R2')
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(iterations, mae_values, marker='o')
    axes[1].set_title('MAE vs Iteration')
    axes[1].set_ylabel('MAE')
    axes[1].grid(True, alpha=0.3)
    
    axes[2].plot(iterations, rmse_values, marker='o')
    axes[2].set_title('RMSE vs Iteration')
    axes[2].set_xlabel('Iteration')
    axes[2].set_ylabel('RMSE')
    axes[2].grid(True, alpha=0.3)
    
    fig.tight_layout()
    plot_filename = os.path.join(
        result_path,
        f"{estimators[strategy_num_int].__class__.__name__}_metrics_plot_{current_time}.png"
    )
    fig.savefig(plot_filename, dpi=200)
    print(f"Metrics plot saved to {plot_filename}")
    plt.close(fig)

if __name__ == "__main__":
    main()
