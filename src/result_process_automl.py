import argparse
import json
# import matplotlib.pyplot as plt
# import numpy as np
import os
import shutil
import numpy as np
import xgboost as xgb
# from sklearn.ensemble import GradientBoostingRegressor
from autosklearn.regression import AutoSklearnRegressor
from autosklearn.pipeline.components.regression import add_regressor
from autosklearn.pipeline.components.base import AutoSklearnRegressionAlgorithm
from autosklearn.pipeline.constants import SPARSE, DENSE, SIGNED_DATA, UNSIGNED_DATA, PREDICTIONS
from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter, UniformIntegerHyperparameter, CategoricalHyperparameter

from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from datetime import datetime

# import tqdm

def _resolve_choice_name(est):
    """
    对 Auto-sklearn 的 *Choice 类* 尽量取被选中的具体组件名。
    兼容：.choice（字符串或对象）、.estimator（对象）、._choice（对象）
    """
    name = est.__class__.__name__
    for attr in ("choice", "estimator", "_choice"):
        if hasattr(est, attr):
            sel = getattr(est, attr)
            if sel is None:
                continue
            return sel if isinstance(sel, str) else sel.__class__.__name__
    return name

def _explode_components(pipeline):
    """
    从 Auto-sklearn 的 Pipeline 中提取三类组件：
    data_preprocessor / feature_preprocessor / regressor
    同时给出底层 sklearn 最终回归器（若能拿到）
    """
    comp = {
        "data_preprocessor": None,
        "feature_preprocessor": None,
        "regressor": None,
        "sklearn_final_regressor": None,  # 可选：底层 sklearn 类名
    }
    if hasattr(pipeline, "steps"):
        for step_name, est in pipeline.steps:
            # 先拿这一层的名称
            step_cls = est.__class__.__name__
            # 若是 Choice 包装，尝试解析被选中者
            if step_cls.endswith("Choice"):
                step_cls = _resolve_choice_name(est)
            comp[step_name] = step_cls

        # 尝试拿到底层 sklearn 回归器（部分组件有 .estimator）
        try:
            last_est = pipeline.steps[-1][1]  # regressor 组件
            # Auto-sklearn 的回归器组件通常有 .estimator 指向 sklearn 对象
            base = getattr(last_est, "estimator", None)
            if base is not None:
                comp["sklearn_final_regressor"] = base.__class__.__name__
        except Exception:
            pass
    return comp


def _pipeline_components_str(model):
    try:
        if hasattr(model, "steps") and isinstance(model.steps, list):
            parts = []
            for name, est in model.steps:
                cls_name = est.__class__.__name__
                # 如果是 Auto-sklearn 的 Choice 类，就进一步深入
                if "Choice" in cls_name:
                    # 通常会有 .choice 或者 .estimator 属性指向实际选择
                    if hasattr(est, "choice") and est.choice is not None:
                        chosen = est.choice.__class__.__name__
                    elif hasattr(est, "estimator") and est.estimator is not None:
                        chosen = est.estimator.__class__.__name__
                    else:
                        chosen = cls_name
                    parts.append(f"{name}:{chosen}")
                else:
                    parts.append(f"{name}:{cls_name}")
            return " -> ".join(parts)
        return model.__class__.__name__
    except Exception:
        return model.__class__.__name__

def _final_estimator_name(model):
    """
    返回 pipeline/集成器里“最终用于回归/分类”的基学习器名称。
    - sklearn Pipeline: 取最后一步(estimator)再递归
    - Voting/Stacking: 展开子估计器名称（已拟合为 estimators_）
    - 其它：回退为类名
    """
    try:
        # sklearn Pipeline: [('preprocessor', ...), ('regressor'/'classifier', Estimator)]
        if hasattr(model, "steps") and isinstance(model.steps, list) and len(model.steps) > 0:
            last_est = model.steps[-1][1]
            return _final_estimator_name(last_est)

        # Voting / Bagging / Stacking（拟合后有 estimators_）
        if hasattr(model, "estimators_"):
            names = []
            for est in getattr(model, "estimators_", []):
                names.append(_final_estimator_name(est))
            # 去重并拼接（避免过长）
            seen, uniq = set(), []
            for n in names:
                if n and n not in seen:
                    uniq.append(n); seen.add(n)
            return f"Voting({'+'.join(uniq)})" if uniq else model.__class__.__name__

        # Stacking 的最终估计器
        if hasattr(model, "final_estimator_"):
            return f"Stacking({_final_estimator_name(model.final_estimator_)})"

        # 普通估计器
        return model.__class__.__name__
    except Exception:
        return model.__class__.__name__

def summarize_autosklearn(automl, iteration_id, dataset_name=None, al_strategy=None):
    # 1) 集成内模型组成（权重）——可用来判定“当轮主导模型”
    ens = automl.get_models_with_weights()  # [(weight, model), ...]
    ens_rows = []
    for w, m in ens:
        # 用 _final_estimator_name 拿到最终基学习器名（而不是 SimpleRegressionPipeline）
        comps = _explode_components(m)
        ens_rows.append({
            'iteration': iteration_id,
            'dataset': dataset_name,
            'al_strategy': al_strategy,
            'ensemble_weight': float(w),

            # 你之前的“成员类型”可以继续用更稳健的最终名（如果你已有 _final_estimator_name 就用它）
            'ensemble_member_type': comps.get('regressor') or m.__class__.__name__,

            # —— 新增三列：管道各组件 ——
            'data_preprocessor': comps['data_preprocessor'],
            'feature_preprocessor': comps['feature_preprocessor'],
            'regressor': comps['regressor'],

            # （可选）底层 sklearn 回归器名
            'sklearn_final_regressor': comps['sklearn_final_regressor'],
        })
    df_ens = pd.DataFrame(ens_rows).sort_values('ensemble_weight', ascending=False)

    # 2) 从 cv_results_ 读“当轮最佳单模型类型 + 其拟合时间（交叉验证平均）”
    cv = pd.DataFrame(automl.cv_results_)
    # 回归任务：键名是 param_regressor:__choice__
    choice_key = 'param_regressor:__choice__' if 'param_regressor:__choice__' in cv.columns else 'param_classifier:__choice__'
    best_rows = cv[cv['rank_test_scores'] == 1]
    top_single_type = best_rows[choice_key].iloc[0]
    top_single_fit_time = float(best_rows['mean_fit_time'].iloc[0])  # 秒

    # 3) 统计当轮“候选评估中各模型类型出现次数 + 各自平均拟合时间”
    by_type = (cv
               .groupby(choice_key)
               .agg(n_runs=('mean_fit_time', 'size'),
                    mean_fit_time=('mean_fit_time', 'mean'))
               .reset_index()
               .rename(columns={choice_key: 'candidate_type'}))
    by_type['iteration'] = iteration_id
    by_type['dataset'] = dataset_name
    by_type['al_strategy'] = al_strategy

    # 4) 精细用时（可选）：从 runhistory_ 拿每次评估的原始耗时
    #    注意：这里的 config -> params 需要用 ids_config 映射
    rh = automl.automl_.runhistory_
    raw_rows = []
    for rk, rv in rh.data.items():
        cfg = rh.ids_config[rk.config_id]
        params = dict(cfg)  # 包含 regressor:__choice__ 等
        raw_rows.append({
            'iteration': iteration_id,
            'dataset': dataset_name,
            'al_strategy': al_strategy,
            'config_id': rk.config_id,
            'candidate_type': params.get('regressor:__choice__', params.get('classifier:__choice__')),
            'status': str(rv.status),
            'cost_internal': rv.cost,
            'time_total': rv.time,                   # 总时间（秒）
            'duration': rv.additional_info.get('duration'),  # 训练时长（秒）
            'starttime': rv.starttime,
            'endtime': rv.endtime,
            # 你也可以把 params 一起存下来做深入分析
        })
    df_runs = pd.DataFrame(raw_rows)

    # 5) 汇总“本轮主导模型类型”（用集成最大权重或最佳单模型二选一）
    dominant = df_ens.iloc[0]['ensemble_member_type'] if len(df_ens) else top_single_type

    summary = {
        'iteration': iteration_id,
        'dataset': dataset_name,
        'al_strategy': al_strategy,
        'dominant_model_by_ensemble': dominant,
        'best_single_model_type': top_single_type,
        'best_single_mean_fit_time_s': top_single_fit_time,
        'n_successful_candidates': int((cv['status'] == 'Success').sum()) if 'status' in cv else len(cv),
    }
    return summary, df_ens, by_type, df_runs


def data_process(file_path, target_columns, target_to_fit, random_state=36):
    # 分割数据集，返回训练集和验证集，重设默认索引
    data = pd.read_csv(file_path)
    data = data.dropna()
    X = data.drop(columns=target_columns)
    y = data[target_to_fit]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
    X_t, X_val, y_t, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=random_state)
    # 重置索引
    X_t = X_t.reset_index(drop=True)  # drop=True 会丢弃旧的索引
    y_t = y_t.reset_index(drop=True)  # 同样丢弃旧的索引

    X_val = X_val.reset_index(drop=True)
    y_val = y_val.reset_index(drop=True)

    return X_t, X_val, y_t, y_val


def data_process_meta(file_path, random_state=36, start_row=0):
    datasets = {}
    meta_data = pd.read_csv(file_path)
    meta_data = meta_data.iloc[start_row:]
    for index, row in meta_data.iterrows():
        target_columns = row['target_columns'].split(';')
        target_to_fit = row['target_to_fit'].split(';')
        X_t, X_val, y_t, y_val = data_process(row['path'], target_columns, target_to_fit, random_state)
        datasets[row['dataname']] = [X_t, X_val, y_t, y_val]
    return datasets


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

class XGBoostRegressor(AutoSklearnRegressionAlgorithm):
    def __init__(self, n_estimators, max_depth, learning_rate, subsample, colsample_bytree, gamma, random_state=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.gamma = gamma
        self.random_state = random_state
        self.estimator = None

    def fit(self, X, y):
        self.n_estimators = int(self.n_estimators)
        self.max_depth = int(self.max_depth)
        self.learning_rate = float(self.learning_rate)
        self.subsample = float(self.subsample)
        self.colsample_bytree = float(self.colsample_bytree)
        self.gamma = float(self.gamma)

        self.estimator = xgb.XGBRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            gamma=self.gamma,
            random_state=self.random_state,
            use_label_encoder=False,
            eval_metric="rmse",
        )
        self.estimator.fit(X, y)
        return self

    def predict(self, X):
        if self.estimator is None:
            raise NotImplementedError
        return self.estimator.predict(X)

    @staticmethod
    def get_properties(dataset_properties=None):
        return {
            "shortname": "XGB",
            "name": "XGBoost Regressor",
            "handles_regression": True,
            "handles_classification": False,
            "handles_multiclass": False,
            "handles_multilabel": False,
            "handles_multioutput": True,
            "is_deterministic": False,
            "input": (SPARSE, DENSE, UNSIGNED_DATA, SIGNED_DATA),
            "output": (PREDICTIONS,),
        }

    @staticmethod
    def get_hyperparameter_search_space(
            feat_type=None, dataset_properties=None
    ):
        cs = ConfigurationSpace()

        n_estimators = UniformIntegerHyperparameter(
            name="n_estimators", lower=50, upper=500, default_value=100
        )
        max_depth = UniformIntegerHyperparameter(
            name="max_depth", lower=3, upper=15, default_value=6
        )
        learning_rate = UniformFloatHyperparameter(
            name="learning_rate", lower=0.01, upper=0.3, default_value=0.1, log=True
        )
        subsample = UniformFloatHyperparameter(
            name="subsample", lower=0.5, upper=1.0, default_value=1.0
        )
        colsample_bytree = UniformFloatHyperparameter(
            name="colsample_bytree", lower=0.5, upper=1.0, default_value=1.0
        )
        gamma = UniformFloatHyperparameter(
            name="gamma", lower=0.0, upper=5.0, default_value=0.0
        )

        cs.add_hyperparameters([n_estimators, max_depth, learning_rate, subsample, colsample_bytree, gamma])
        return cs

def data_extraction(idx, X):
    # 使用 .iloc 提取特定索引的行
    extracted_data = X.loc[idx]
    # 使用 .drop 删除这些行，注意设置 inplace=False 以返回新的 DataFrame
    remaining_data = X.drop(idx)
    return extracted_data, remaining_data

# 自定义转换函数
def convert_to_serializable(obj):
    if isinstance(obj, np.int64):  # 如果是 int64 类型
        return int(obj)  # 转换为 Python 的 int 类型
    raise TypeError(f"Type {type(obj)} not serializable")  # 其他类型抛出错误

if __name__ == '__main__':
    add_regressor(XGBoostRegressor)
    parser = argparse.ArgumentParser(description="Run AutoSklearn on a specific dataset.")
    parser.add_argument("--random_state", type=int, required=True, help="Random state for fitting.")
    parser.add_argument("--initial-method", type=str, default='random', help="Initial method for active learning.")
    parser.add_argument("--strategy-idx", type=int, required=True, help="ID of the strategy to fit.")
    parser.add_argument("--dataset-idx", type=int, required=True, help="ID of the dataset to fit.")
    parser.add_argument("--time-limit", type=int, default=300, help="Time limit for fitting in seconds.")
    parser.add_argument("--n_pro_query", type=int, default=10, help="Number of queries per iteration.")
    args = parser.parse_args()

    with open('strategy_name.json', 'r') as f:
        strategy_name_list = json.load(f)

    with open('dataset_name.json', 'r') as f:
        dataset_name_list = json.load(f)

    random_state = args.random_state
    initial_method = args.initial_method
    strategy_name = strategy_name_list[args.strategy_idx]
    dataset_name = dataset_name_list[args.dataset_idx]
    n_pro_query = args.n_pro_query

    tmp_dir = '/data/horse/ws/jibi984b-al/tmp'
    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir, exist_ok=True)


    # output_dir = './result_automl'
    # if not os.path.exists(output_dir):
    #     os.makedirs(output_dir, exist_ok=True)


    result_dir = '../result_2'
    result_dir = os.path.join(result_dir, f'{n_pro_query}_pro_query')
    result_dir = os.path.join(result_dir, f'{random_state}')
    result_dir = os.path.join(result_dir, f'{initial_method}')

    result_files = os.listdir(result_dir)
    # result_files = [f for f in result_files if f.endswith('.json')]
    result_file_name = None
    result_path = None
    for f in result_files:
        # if strategy_name == 'BMDAL_lcmd':
        #     strategy_name = 'BMDAL'
        if f.startswith(f'{strategy_name}_{dataset_name}'):
            result_path = os.path.join(result_dir, f)
            result_file_name = f
            break

    if result_file_name is None:
        print(f'No such file: {strategy_name}_{dataset_name}')
        exit(1)


    # 打开result_files[0]文件
    with open(result_path, 'r') as f:
        result_query_id = json.load(f)

    # datasets = data_process_meta('../dataset/meta.csv', random_state=36)

    # strategies_name = []
    # datasets_name = []
    # for list_item in result:
    #     if list(list_item.keys())[0] not in strategies_name:
    #         strategies_name.append(list(list_item.keys())[0])
    #     if list_item['dataset_name'] not in datasets_name:
    #         datasets_name.append(list_item['dataset_name'])
    #
    #
    # results = {}
    # for file in result_files:
    #     with open(os.path.join(result_dir, file), 'r') as f:
    #         result = json.load(f)
    #     results[result[0]['random_state']] = result

    result_dict = {}
    datasets = data_process_meta('../dataset/meta.csv', random_state=random_state)
    X_labeled = pd.DataFrame()
    y_labeled = pd.DataFrame()
    X_t, X_val, y_t, y_val = datasets[dataset_name]
    # print(X_t.shape, X_val.shape, y_t.shape, y_val.shape)
    X_unlabeled = X_t.copy()
    y_unlabeled = y_t.copy()
    query_id_list = result_query_id[strategy_name]
    mse_list = []
    mae_list = []
    rmse_list = []
    r2_list = []
    top_model_dict = {}

    summaries, ensembles, candidates, runhistories = [], [], [], []
    for idx, query_id in enumerate(query_id_list):

        if query_id == []:
            continue

        # 获取当前时间戳
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        task_tmp_folder = os.path.join(tmp_dir, f'{random_state}_{initial_method}_{strategy_name}_{dataset_name}_{idx}_{n_pro_query}_{current_time}')

        # 如果task_tmp_folder存在，删除
        # if os.path.exists(task_tmp_folder):
        #     os.rmdir(task_tmp_folder)

        # task_output_folder = os.path.join(output_dir, f'{random_state}_{initial_method}_{strategy_name}_{dataset_name}_{idx}')
        # if not os.path.exists(task_output_folder):
        #     os.makedirs(task_output_folder, exist_ok=True)



        X_query, X_unlabeled = data_extraction(query_id, X_unlabeled)
        y_query, y_unlabeled = data_extraction(query_id, y_unlabeled)
        # print(X_query.shape, X_unlabeled.shape, y_query.shape, y_unlabeled.shape)
        # print(type(X_query), type(X_unlabeled), type(y_query), type(y_unlabeled))
        X_labeled = pd.concat([X_labeled, X_query])
        y_labeled = pd.concat([y_labeled, y_query])
        # print(X_labeled.shape, y_labeled.shape)
        model = AutoSklearnRegressor(
            time_left_for_this_task=args.time_limit,  # 总时间限制 2 小时
            per_run_time_limit=args.time_limit/10,  # 每次拟合限制 5 分钟
            n_jobs=-1,  # 使用所有可用线程
            seed=random_state,  # 设置随机种子
            resampling_strategy='cv',
            resampling_strategy_arguments={'folds': 5},
            delete_tmp_folder_after_terminate=True,
            tmp_folder=task_tmp_folder,
        )

        model.fit(X_labeled, y_labeled)

        summary, df_ens, by_type, df_runs = summarize_autosklearn(model, iteration_id=idx, dataset_name=dataset_name, al_strategy=strategy_name)

        summaries.append(summary)
        ensembles.append(df_ens)
        candidates.append(by_type)
        runhistories.append(df_runs)

        y_pred = model.predict(X_val)
        mse = mean_squared_error(y_val, y_pred)
        mae = mean_absolute_error(y_val, y_pred)
        rmse = custom_rmse(y_val, y_pred)
        r2 = r2_score(y_val, y_pred)
        mse_list.append(mse)
        mae_list.append(mae)
        rmse_list.append(rmse)
        r2_list.append(r2)
        # 获得模型集合的权重
        models_with_weights = model.get_models_with_weights()
        top_5_models = sorted(models_with_weights, key=lambda x: x[0], reverse=True)[:5]
        top_5_models_dict = [
            {
                "rank": rank,
                "weight": weight,
                "model": str(model)  # 将模型转换为字符串表示
            }
            for rank, (weight, model) in enumerate(top_5_models, start=1)
        ]
        # result['top_3_models'] = top_5_models_dict

        top_model_dict[idx] = top_5_models_dict
        if os.path.exists(task_tmp_folder):
            shutil.rmtree(task_tmp_folder)
        if model.dask_client is not None:
            model.dask_client.close()
            print('close dask client')

    result_dict['strategy_name'] = strategy_name
    result_dict['dataset_name'] = dataset_name
    result_dict['random_state'] = random_state
    result_dict['mse'] = mse_list
    result_dict['mae'] = mae_list
    result_dict['rmse'] = rmse_list
    result_dict['r2'] = r2_list

    # 保存automl总结结果
    summary_table   = pd.DataFrame(summaries)
    ensemble_table  = pd.concat(ensembles, ignore_index=True) if ensembles else pd.DataFrame()
    candidates_table= pd.concat(candidates, ignore_index=True) if candidates else pd.DataFrame()
    runhistory_table= pd.concat(runhistories, ignore_index=True) if runhistories else pd.DataFrame()
    result_dir = os.path.join(result_dir, 'automl')

    new_automl_dir = os.path.join(result_dir, 'new_automl')
    new_automl_dir = os.path.join(new_automl_dir, strategy_name, dataset_name, f"{args.time_limit}s_time_limit")

    log_automl_dir = os.path.join(result_dir, 'logs')
    log_automl_dir = os.path.join(log_automl_dir, strategy_name, dataset_name, f"{args.time_limit}s_time_limit")

    os.makedirs(result_dir, exist_ok=True)
    os.makedirs(log_automl_dir, exist_ok=True)
    os.makedirs(new_automl_dir, exist_ok=True)
    # 保存结果表格
    summary_table.to_csv(os.path.join(log_automl_dir, 'autosklearn_summary.csv'), index=False)
    ensemble_table.to_csv(os.path.join(log_automl_dir, 'autosklearn_ensemble_members.csv'), index=False)
    candidates_table.to_csv(os.path.join(log_automl_dir, 'autosklearn_candidates_by_type.csv'), index=False)
    runhistory_table.to_csv(os.path.join(log_automl_dir, 'autosklearn_runhistory_raw.csv'), index=False)

    # result_file_name字符串前面加上automl_
    result_automl_file_name = 'automl_' + result_file_name
    result_automl_model_record_file_name = 'model_record_' + result_automl_file_name



    with open(os.path.join(new_automl_dir, result_automl_file_name), 'w') as f:
        json.dump(result_dict, f)

    with open(os.path.join(new_automl_dir, result_automl_model_record_file_name), 'w') as f:
        json.dump(top_model_dict, f, default=convert_to_serializable)