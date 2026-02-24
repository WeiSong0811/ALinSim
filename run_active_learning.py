import random
import argparse
import numpy as np
import pandas as pd
import torch
# from matplotlib import pyplot as plt
from sklearn.gaussian_process.kernels import RBF, Matern, ConstantKernel as C, WhiteKernel
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.backends.opt_einsum import strategy
from tqdm import tqdm
import json
from utils import data_process, data_process_yin, data_process_meta, active_learning
from strategies import TreeBasedRegressor_Representativity_self
import warnings
# from config_loader import load_config, create_kernel
# from sklearn.ensemble import RandomForestRegressor
import pickle
from datetime import datetime
import os

# 忽略所有警告
warnings.filterwarnings("ignore")
# # 加载配置文件
# config = load_config()

# initial_method = "random"  # greedy_search kmeans ncc random
test_method = ["normal",
               # "LOO_k-Fold-CV",
               "k-Fold-CV",
               "RSValidation"]

parser = argparse.ArgumentParser(description="Run AutoSklearn on a specific dataset.")
parser.add_argument("--random_state", type=int, required=True, help="Random state for fitting.")
parser.add_argument("--initial-method", type=str, default='random', help="Initial method for active learning.")
parser.add_argument("--strategy-idx", type=int, required=True, help="ID of the strategy to fit.")
parser.add_argument("--dataset-idx", type=int, required=True, help="ID of the dataset to fit.")
parser.add_argument("--n_pro_query", type=int, default=10, help="Number of queries per iteration.")
# parser.add_argument("--time-limit", type=int, default=3600, help="Time limit for fitting in seconds.")
args = parser.parse_args()
random_state = args.random_state
initial_method = args.initial_method
strategy_num_int = args.strategy_idx
datasets_num_int = args.dataset_idx
n_pro_query = args.n_pro_query

# # kernel = create_kernel(config)
# kernel = RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e3)) + WhiteKernel(noise_level=1,
#                                                                               noise_level_bounds=(1e-10, 1e+1))
# kernel = (C(1.0, (1e-2, 1e2)) * RBF(1.0, (1e-2, 1e2)) +
#           WhiteKernel(noise_level=1, noise_level_bounds=(1e-10, 1e+1)))
threshold = 0.85 
# 打印验证加载的参数
print(f"Random State: {random_state}")
print(f"Initial Method: {initial_method}")
# print(f"Kernel: {kernel}")
# print(f"Threshold: {threshold}")

current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


# 设置随机种子
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(random_state)
# 定义主动学习策略
estimators = [TreeBasedRegressor_Representativity_self(random_state=random_state, min_samples_leaf=5)]

# if not os.path.exists(f"../result/random_state_{random_state}_{initial_method}"):
#     os.makedirs(f"../result/random_state_{random_state}_{initial_method}")

datasets = data_process_meta('../dataset/meta.csv', random_state=random_state)

result_path = f"../result_review/"
result_path = os.path.join(result_path, str(n_pro_query), str(random_state), initial_method)

result_time_record_path = os.path.join(result_path, 'time_record')
if not os.path.exists(result_time_record_path):
    os.makedirs(result_time_record_path, exist_ok=True)

if not os.path.exists(result_path):
    os.makedirs(result_path, exist_ok=True)
    
print(f'result_path: {result_path}')
print(f'result_time_record_path: {result_time_record_path}')
results = {}


datasets_list = list(datasets.items())
key = datasets_list[datasets_num_int][0]
value = datasets_list[datasets_num_int][1]
estimators_list = [estimators[strategy_num_int]]

print('*' * 15, f'Processing {key} dataset')
X_t, X_val, y_t, y_val = value

duplicate_samples = X_t.duplicated()
# 检查X_t的数据量
total_data_volume = X_t.shape[0]
n_initial = 10
# n_pro_query = 10
n_queries = (total_data_volume - n_initial - 1) // n_pro_query
# n_queries = 5
print(f"Total data volume: {total_data_volume}, n_initial: {n_initial}, n_pro_query: {n_pro_query}, "
      f"n_queries: {n_queries}")

result_filename = f"/{estimators[strategy_num_int].__class__.__name__}_{key}"
print(f"Result filename: {result_filename}")


query_idx_all, query_time_all= active_learning(estimators_list, X_t, y_t, X_val, y_val, n_initial,
                                                                   n_pro_query, n_queries, threshold,
                                                                   initial_method=initial_method,
                                                                   test_methods=test_method,
                                                                   random_state=random_state)

result = query_idx_all.copy()
result['dataset_name'] = key
result['initial_method'] = initial_method
result['random_state'] = random_state

result_time_record = query_time_all.copy()
result_time_record['dataset_name'] = key
result_time_record['initial_method'] = initial_method
result_time_record['random_state'] = random_state

# 保存结果到json文件

result_json_filename = os.path.join(result_path, f"{estimators[strategy_num_int].__class__.__name__}_{key}_{current_time}.json")
with open(result_json_filename, 'w', encoding='utf-8') as f:
    json.dump(result, f)
    print(f"Dictionary saved to {result_json_filename}")

result_time_record_json_filename = os.path.join(result_time_record_path, f"{estimators[strategy_num_int].__class__.__name__}_{key}_{current_time}.json")
with open(result_time_record_json_filename, 'w', encoding='utf-8') as f:
    json.dump(result_time_record, f)
    print(f"Dictionary saved to {result_time_record_json_filename}")