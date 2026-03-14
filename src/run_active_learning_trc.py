import random
import argparse
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import json
from utils import active_learning_trc, generation_pool_trc
from run_trc import predict_with_retry
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
    # datasets_num_int = args.dataset_idx
    # n_pro_query = args.n_pro_query
    n_pro_query = 5
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
    
    _ , X_pool_filtered,_ = generation_pool_trc(seed=random_state)
    
    # 创建输出文件夹
    result_path = f"../results/result_single_trc/"
    result_path = os.path.join(result_path, str(random_state))
    
    result_time_record_path = os.path.join(result_path, 'time_record')
    result_xy_record_path = os.path.join(result_path, 'xy_record', current_time)
    if not os.path.exists(result_time_record_path):
        os.makedirs(result_time_record_path, exist_ok=True)
    
    if not os.path.exists(result_path):
        os.makedirs(result_path, exist_ok=True)
    if not os.path.exists(result_xy_record_path):
        os.makedirs(result_xy_record_path, exist_ok=True)
        
    print(f'result_path: {result_path}')
    print(f'result_time_record_path: {result_time_record_path}')
    print(f'result_xy_record_path: {result_xy_record_path}')
    
    results = {}
    
    # datasets_list = list(X_pool_filtered.items())
    # key = datasets_list[0]
    # value = datasets_list[1]
    
    estimators_list = [estimators[strategy_num_int]]
    
    # print('*' * 15, f'Processing {key} dataset')

    # target_variable = ['f_res']
    X_t = X_pool_filtered
    # fea_df = pd.read_csv(f'./run_trc/predicted_datasets/sim_trc_{random_state}.csv')
    # X_test = fea_df.drop(columns=target_variable)
    # y_val = fea_df[target_variable]
    # X_val = X_test.copy()
    # y_val = pd.DataFrame(y_val, columns=target_variable)
    # y_val_out = pd.concat([X_test.reset_index(drop=True), y_val], axis=1)
    # y_val_out.to_csv(f'../data/pan_test.csv', index=False)
    # X_t, X_val, y_t, y_val = value
    
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
    
    query_idx_all, query_time_all = active_learning_trc(estimators_list, X_t, n_initial,
                                                                       n_pro_query, n_queries,
                                                                       random_state=random_state,
                                                                       save_dir=result_xy_record_path)
    
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
    
if __name__ == "__main__":
    main()
