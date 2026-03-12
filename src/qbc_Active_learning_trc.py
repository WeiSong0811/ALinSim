import argparse

from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error, root_mean_squared_error
import pandas as pd
from strategies import QBC_Paper
from utils import generation_pool_trc
from run_trc import predict
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
'''
def simulator(X_single_row):

    ######################## 内部忽略，就是个仿真程序的接口，输入参数X，输出目标变量y ####################################
    """
    输入特征向量 X (1D array), 返回对应的 y。
    使用 np.isclose 处理浮点数精度问题，确保能匹配上原始数据集。
    """
    # 强制转换为 numpy 以防万一
    X_single_row = np.array(X_single_row)
    
    # 找到原始特征列
    feature_cols = Parameter_space.columns
    original_features = data[feature_cols].values
    
    # 计算每一行与输入 X 的差异
    # np.isclose 能处理像 0.300000000004 vs 0.3 这样的微小误差
    mask = np.all(np.isclose(original_features, X_single_row, atol=1e-8), axis=1)
    
    y_simulated = data.loc[mask, target_variable].values
    
    if len(y_simulated) == 0:
        raise ValueError(f"无法在数据集中找到特征值为 {X_single_row} 的样本，请检查输入是否经过了未还原的缩放。")
        
    #################################################################################################################

    return y_simulated[0]
'''

def dataset_generation(idx_list):

    query_X_df = Parameter_space.loc[idx_list].copy()
    y_batch = np.asarray(predict(query_X_df.values)).reshape(-1, 1)
    query_y_df = pd.DataFrame(y_batch, index=idx_list, columns=target_variable)
    return query_X_df, query_y_df

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
query_size = 2
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
        query_idx_list = qbc_active_learner.query(X_unlabeled=X_pool_scaled, n_act=query_size, X_labeled=X_train, y_labeled=y_train)

    query_X_df, query_y_df = dataset_generation(query_idx_list)

    Parameter_space = Parameter_space.drop(index=query_idx_list)  # 从Parameter_space中删除已查询的样本

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






