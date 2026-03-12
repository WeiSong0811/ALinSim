import argparse

from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel as C
from sklearn.metrics import r2_score, mean_absolute_error, root_mean_squared_error
import pandas as pd
from strategies import GaussianProcessBased
from utils import generation_pool_trc
from run_trc import predict_with_retry
import json
import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description="Run GP_AL on a specific dataset.")
parser.add_argument("--random_state", type=int, required=True, help="Random state for fitting.")
# parser.add_argument("--label_idx", type=int, default=0, help="Index of the target variable to fit.")
# parser.add_argument("--time-limit", type=int, default=3600, help="Time limit for fitting in seconds.")
args = parser.parse_args()
# label_idx = args.label_idx
seed = args.random_state

# data = pd.read_csv('data/concrete_data.csv')  # Replace with your dataset path
# target_variable = 'concrete_compressive_strength'  # Replace with your target variable name
target_variable = ['f_res']
_ , X_pool_filtered, _ = generation_pool_trc(seed=seed)

# inpute test data
fea_df = pd.read_csv(f'./run_trc/predicted_datasets/sim_trc_{seed}.csv')
x_test = fea_df.drop(columns=target_variable)
y_test = fea_df[target_variable]

x_train = X_pool_filtered.copy() # 参数空间

global Parameter_space
global y
# X, y = data.drop(columns=[target_variable]), data[target_variable]

# x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=150, random_state=seed)

Parameter_space = x_train.copy()  # 将训练集的特征作为Parameter_space，后续从中查询样本

d = Parameter_space.shape[1]  # 你的特征维度

kernel_RBF = (
    C(1.0, (1e-3, 1e3)) *
    RBF(length_scale=np.ones(d), length_scale_bounds=(1e-2, 1e2))
    +
    WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-8, 1e0))
)

GP_active_learner = GaussianProcessBased(kernel=kernel_RBF, n_restarts_optimizer=15, random_state=seed)

query_steps = 20
query_size = 5
initial_query_size = 5

mae_list = []
rmse_list = []
r2_list = []

results_fea = []
for step in range(query_steps+1):
    print(f'Query step: {step}')

    if step == 0:
        # 在Parameter_space中随机选择10个样本的索引作为初始训练集
        rng = np.random.default_rng(seed)
        initial_idx_list = rng.choice(Parameter_space.index, size=initial_query_size, replace=False).tolist()
        print('step == 0', initial_idx_list)
        query_idx_list = initial_idx_list
    else:
        X_pool_scaled = pd.DataFrame(
            scaler.transform(Parameter_space.values),
            index=Parameter_space.index,
            columns=Parameter_space.columns
        )
        query_idx_list = GP_active_learner.query(X_unlabeled=X_pool_scaled, n_act=3*query_size)
        
    candidate_X = Parameter_space.loc[query_idx_list].values

    valid_y, consumed_idx, valid_idx = predict_with_retry(
        X_candidates=candidate_X,
        idx_candidates=query_idx_list,
        target_valid=5,
        init_parallel=5,
        seed=seed
    )
    query_X_df = Parameter_space.loc[valid_idx].copy()
    query_y_df = pd.DataFrame(valid_y, index=valid_idx, columns=target_variable)
    # query_X_df, query_y_df = dataset_generation(query_idx_list)
    # query_X_df, query_y_df, consumed_idx, unused_idx = dataset_generation_with_retry(query_idx_list, target_valid=query_size)

    print(f'{step}:',query_X_df, query_y_df)

    Parameter_space = Parameter_space.drop(index=consumed_idx)  # 从Parameter_space中删除已查询的样本

    if step == 0:
        X_train = query_X_df.values
        y_train = query_y_df.values
    else:
        X_train = np.vstack((X_train, query_X_df.values))
        y_train = np.vstack((y_train, query_y_df.values))
    print(f'{step}: X_train und x_test: ',X_train, y_train)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    x_test_scaled = scaler.transform(x_test)
    results_fea.append(
        {
            'Step': step,
            'X_train': X_train_scaled.copy().tolist(),
            'y_train': y_train.copy().tolist()
        }
    )
    GP_active_learner.fit(X_train_scaled, y_train)
    y_pred = GP_active_learner.predict(x_test_scaled)

    mae_list.append(mean_absolute_error(y_test, y_pred))
    rmse_list.append(root_mean_squared_error(y_test, y_pred))
    r2_list.append(r2_score(y_test, y_pred))

# 保存结果到json
results = {
    "mae": mae_list,
    "rmse": rmse_list,
    "r2": r2_list
}
with open(f'../results/result_single_trc/GP_AL_results_seed_{seed}.json', 'w') as f:
    json.dump(results, f)

with open(f'../results/result_single_trc/GP_AL_results_seed_{seed}_fea.json', 'w') as f:
    json.dump(results_fea, f)
