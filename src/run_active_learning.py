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
from strategies import (TreeBasedRegressor_Diversity, TreeBasedRegressor_Representativity, GaussianProcessBased,
                        QueryByCommittee, Basic_RD_ALR, GSBAG, QDD, RandomSearch, GSi, LearningLoss, BMDAL,
                        RD_GS_ALR, RD_QBC_ALR, RD_EMCM_ALR, EGAL, GSx, GSy, mcdropout,
                        TreeBasedRegressor_Diversity_self,
                        TreeBasedRegressor_Representativity_self)
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

# 获取当前脚本的文件名（包括路径）
# script_path = __file__

# 仅获取文件名，不包括路径
# script_name = os.path.basename(script_path).split('.')[0]
# if 'main' not in script_name:
#     script_name_list = script_name.split('_')
#     strategy_num = script_name_list[0]
#     initial_method_num = script_name_list[1]
#     datasets_num = script_name_list[2]
#     random_state_num = script_name_list[3]
#     # 转换为整数
#     strategy_num_int = int(strategy_num)
#     datasets_num_int = int(datasets_num)
#     random_state = int(random_state_num)
#     # 获取文件名中数字部分
#     # greedy_search kmeans ncc random
#     if initial_method_num == 'r':
#         initial_method = 'random'
#     elif initial_method_num == 'g':
#         initial_method = 'greedy_search'
#     elif initial_method_num == 'k':
#         initial_method = 'kmeans'
#     elif initial_method_num == 'n':
#         initial_method = 'ncc'
#     else:
#         raise ValueError("Invalid initial method")
# else:
#     strategy_num = 0
#     initial_method = 'random'
#     datasets_num = 0
#     random_state = 0

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
kernel = (C(1.0, (1e-2, 1e2)) * RBF(1.0, (1e-2, 1e2)) +
          WhiteKernel(noise_level=1, noise_level_bounds=(1e-10, 1e+1)))
threshold = 0.85
# 打印验证加载的参数
print(f"Random State: {random_state}")
print(f"Initial Method: {initial_method}")
# print(f"Kernel: {kernel}")
print(f"Threshold: {threshold}")

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
# bmdal methode： 'maxdiag', 'maxdet', 'bait', 'fw' (for FrankWolfe), 'maxdist', 'kmeanspp', 'lcmd'
# bmdal kernel：  'll', 'grad', 'lin', 'nngp', 'ntk', and 'laplace'.

# 定义主动学习策略
estimators = [
    RandomSearch(random_state=random_state),
    GSBAG(random_state=random_state, kernel=kernel, n_restarts_optimizer=10),
    QueryByCommittee(random_state=random_state, num_learner=100),
    # TreeBasedRegressor_Diversity(random_state=random_state, min_samples_leaf=5),
    # TreeBasedRegressor_Representativity(random_state=random_state, min_samples_leaf=5),
    TreeBasedRegressor_Diversity_self(random_state=random_state, min_samples_leaf=5),
    TreeBasedRegressor_Representativity_self(random_state=random_state, min_samples_leaf=5),
    GaussianProcessBased(random_state=random_state, kernel=kernel, n_restarts_optimizer=10),
    QDD(random_state=random_state),
    GSi(random_state=random_state),
    GSx(random_state=random_state),
    GSy(random_state=random_state),
    LearningLoss(BATCH=16, LR=0.01, MARGIN=1, WEIGHT=0.0001, EPOCH=200, EPOCHL=75, WDECAY=5e-4, random_state=random_state),
    EGAL(b_factor=0.25, random_state=random_state),
    mcdropout(random_state=random_state,learning_rate=0.01,num_epochs=200,batch_size=16),
    Basic_RD_ALR(random_state=random_state),
    RD_GS_ALR(random_state=random_state),
    RD_QBC_ALR(random_state=random_state, num_learner=100),
    RD_EMCM_ALR(random_state=random_state),
    BMDAL(random_state=random_state, selection_method='lcmd'),
    # BMDAL(random_state=random_state, selection_method='maxdiag'),
    # BMDAL(random_state=random_state, selection_method='maxdet'),
    # BMDAL(random_state=random_state, selection_method='bait'),
    # BMDAL(random_state=random_state, selection_method='fw'),
    # BMDAL(random_state=random_state, selection_method='maxdist'),
    # BMDAL(random_state=random_state, selection_method='kmeanspp'),
]

# if not os.path.exists(f"../result/random_state_{random_state}_{initial_method}"):
#     os.makedirs(f"../result/random_state_{random_state}_{initial_method}")

datasets = data_process_meta('../dataset/meta.csv', random_state=random_state)

# result_path = f"../result/12_09_new_random_state_{random_state}_{initial_method}"
# result_path = f"../result_2/{n_pro_query}_pro_query/{random_state}/{initial_method}"
result_path = f"../result_review/"
result_path = os.path.join(result_path, str(n_pro_query), str(random_state), initial_method)

result_time_record_path = os.path.join(result_path, 'time_record')
if not os.path.exists(result_time_record_path):
    os.makedirs(result_time_record_path, exist_ok=True)

if not os.path.exists(result_path):
    os.makedirs(result_path, exist_ok=True)
    
print(f'result_path: {result_path}')
print(f'result_time_record_path: {result_time_record_path}')
# ['yin-pullout1', 'yin-pullout2', 'concrete_slump_test', 'uci-concrete', 'bfrc_cs', 'bfrc_fs', 'bfrc_sts',
# 'uhpc_cs', 'uhpc_fs', 'uhpc_ss', 'uhpc_p']
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

# # sub_pickle_filename = f"/{estimators[strategy_num_int].__class__.__name__}_{key}_{current_time}.json"
# sub_pickle_filename = result_path + f"/{estimators[strategy_num_int].__class__.__name__}_{key}_{current_time}.json"
# print(f"Sub pickle filename: {sub_pickle_filename}")
# files = os.listdir(result_path)
# print(f'lenth of files: {len(files)}')
#
# for file in files:
#     print(file)
#     if file.endswith('.json') and result_filename in file:
#         print(f"File {file} already exists, skipping...")
#         exit(0)



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

# results[key] = result

# results['random_state'] = random_state

# pickle_filename = result_path + f"/data_{current_time}_random_states({random_state})_{initial_method}_{strategy_num}.json"
# pickle_strategies_instance_list = result_path + f"/data_{current_time}_random_states({random_state})_{initial_method}_{strategy_num}_strategies_instance_list.json"

# with open(pickle_filename, 'w', encoding='utf-8') as json_file:
#     json.dump(results, json_file)
#     print(f"Dictionary saved to {pickle_filename}")

# with open(pickle_strategies_instance_list, 'w', encoding='utf-8') as json_file:
#     json.dump(estimators, json_file)
#     print(f"Dictionary saved to {pickle_strategies_instance_list}")
