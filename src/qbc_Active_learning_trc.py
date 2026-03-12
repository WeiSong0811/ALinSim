import argparse

from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error, root_mean_squared_error
import pandas as pd
from strategies import QBC_Paper
from utils import generation_pool_trc
from run_trc import predict_with_retry
import warnings
warnings.filterwarnings("ignore")
from xgboost import XGBRegressor
parser = argparse.ArgumentParser(description="Run GP_AL on a specific dataset.")
parser.add_argument("--random_state", type=int, required=True, help="Random state for fitting.")
parser.add_argument("--label_idx", type=int, default=0, help="Index of the target variable to fit.")
# parser.add_argument("--time-limit", type=int, default=3600, help="Time limit for fitting in seconds.")
args = parser.parse_args()
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

# kernel_RBF = (
#     C(1.0, (1e-3, 1e3)) *
#     RBF(length_scale=np.ones(d), length_scale_bounds=(1e-2, 1e2))
#     +
#     WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-8, 1e0))
# )

# GP_active_learner = GaussianProcessBased(kernel=kernel_RBF, n_restarts_optimizer=15, random_state=seed)

qbc_active_learner = QBC_Paper(random_state=seed)

query_steps = 20
query_size = 5
initial_query_size = 5

mae_list = []
rmse_list = []
r2_list = []

model = XGBRegressor(
    booster="gbtree",
    n_estimators=1000,
    learning_rate=0.1,
    reg_alpha=0.01,      # L1
    reg_lambda=0.1,      # L2
    tree_method="hist",
    subsample=0.85,
    colsample_bytree=0.3,
    colsample_bylevel=0.5,
    random_state=seed,
    n_jobs=-1
)

for step in range(query_steps+1):
    print(f'Query step: {step}')

    if step == 0:
        # 在Parameter_space中随机选择10个样本的索引作为初始训练集
        rng = np.random.default_rng(seed)
        initial_idx_list = rng.choice(Parameter_space.index, size=initial_query_size, replace=False).tolist()
        query_idx_list = initial_idx_list
    else:
        X_pool_scaled = pd.DataFrame(
            scaler.transform(Parameter_space.values),
            index=Parameter_space.index,
            columns=Parameter_space.columns
        )
        query_idx_list = qbc_active_learner.query(X_unlabeled=X_pool_scaled, n_act=3 * query_size, X_labeled=X_train, y_labeled=y_train)

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

    Parameter_space = Parameter_space.drop(index=consumed_idx)  # 从Parameter_space中删除已查询的样本

    if step == 0:
        X_train = query_X_df.values
        y_train = query_y_df.values
    else:
        X_train = np.vstack((X_train, query_X_df.values))
        y_train = np.vstack((y_train, query_y_df.values))

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    x_test_scaled = scaler.transform(x_test)
    
    model.fit(X_train_scaled, y_train)
    print('in Step:', step, 'X train shape:',X_train_scaled.shape)
    y_pred = model.predict(x_test_scaled)

    mae_list.append(mean_absolute_error(y_test, y_pred))
    rmse_list.append(root_mean_squared_error(y_test, y_pred))
    r2_list.append(r2_score(y_test, y_pred))

# 保存结果到json
import json


results = {
    "mae": mae_list,
    "rmse": rmse_list,
    "r2": r2_list
}

with open(f'../results/result_single_trc/QBC_AL_PAN_results_seed_{seed}.json', 'w') as f:
    json.dump(results, f)






