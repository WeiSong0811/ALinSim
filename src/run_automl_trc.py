import os
import numpy as np
import pandas as pd
import json
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


def run_active_learning_with_absolute_idx(
    taskname: str,
    seed: int,
    target_col: str,
    idx_list: list,   # 例如 [[1,5,7,9,11], [52,26], [16,87], ...]
    use_scaling: bool = True,
    verbose: bool = True,
    step_idx: int = 0,
):
    """
    主动学习训练循环（适配绝对idx）
    
    参数
    ----
    taskname : str
        任务名，用于读取CSV文件
    seed : int
        随机种子（用于文件名）
    target_col : str
        目标列名
    idx_list : list[list[int]]
        每一轮新增采样的“绝对idx”列表
        例如:
        [
            [1,5,7,9,11],   # 初始样本
            [52,26],        # 第1轮新增
            [16,87],        # 第2轮新增
            ...
        ]
    autosklearn : object
        具有 fit(X, y) 和 predict(X) 的回归模型（例如 AutoSklearnRegressor）
    use_scaling : bool
        是否进行标准化（GP/SVR/MLP建议True；树模型可False）
    verbose : bool
        是否打印每轮信息

    返回
    ----
    results_df : pd.DataFrame
        每轮评估结果表
    labeled_idx_abs : list[int]
        最终累计已标注绝对idx
    """

    # ========= 1) 读取数据 =========
    train_path = f"{taskname}_{seed}_train.csv"
    test_path = f"{taskname}_{seed}_test.csv"

    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)

    # idx 是“绝对 idx”
    # 所以 CSV 里必须保留原始 idx（作为一列），这里把它设为 DataFrame index
    # 假设这一列名叫 "idx"；如果列名不同，改这里
    if "idx" not in train_data.columns or "idx" not in test_data.columns:
        raise ValueError(
            "train/test CSV 中必须包含绝对索引列 'idx'。"
            "如果你的列名不是 idx，请修改代码中的 set_index('idx') 部分。"
        )

    train_data = train_data.set_index("idx")
    test_data = test_data.set_index("idx")

    # 检查目标列
    if target_col not in train_data.columns or target_col not in test_data.columns:
        raise ValueError(f"target_col='{target_col}' 不在 train/test CSV 列中。")

    # 拆分 X/y（保留绝对idx作为 index）
    X_train_raw = train_data.drop(columns=[target_col])
    y_train = train_data[target_col]

    X_test_raw = test_data.drop(columns=[target_col])
    y_test = test_data[target_col]

    # ========= 2) 基本集合 =========
    train_idx_set = set(X_train_raw.index.tolist())
    test_idx_set = set(X_test_raw.index.tolist())

    # 防御性检查：train/test 不应重叠
    overlap_train_test = train_idx_set & test_idx_set
    if len(overlap_train_test) > 0:
        raise ValueError(
            f"训练集和测试集索引重叠，存在数据泄漏！重叠数量={len(overlap_train_test)}，"
            f"示例={list(sorted(overlap_train_test))[:10]}"
        )

    # ========= 3) 主动学习循环 =========
    labeled_idx_abs = []   # 累计已标注绝对idx

    # for round_id, new_idx_abs in enumerate(idx_list):
    
    round_id = step_idx

    for i in range(round_id + 1):
        labeled_idx_abs.extend(idx_list[i])



    # 当前累计训练集（用 .loc，因为是绝对idx）
    X_labeled = X_train_raw.loc[labeled_idx_abs]
    y_labeled = y_train.loc[labeled_idx_abs]

    # 预处理（每轮重新 fit，避免泄漏）
    if use_scaling:
        scaler = StandardScaler()
        X_labeled_proc = scaler.fit_transform(X_labeled)
        X_test_proc = scaler.transform(X_test_raw)
    else:
        # 树模型可不标准化
        X_labeled_proc = X_labeled.values
        X_test_proc = X_test_raw.values

    # 训练模型（累计数据）
    autosklearn.fit(X_labeled_proc, y_labeled.values)

    # 测试集评估（固定测试集）
    y_pred = autosklearn.predict(X_test_proc)

    r2 = r2_score(y_test.values, y_pred)
    mae = mean_absolute_error(y_test.values, y_pred)
    rmse = mean_squared_error(y_test.values, y_pred) ** 0.5

    results = {
        "round": round_id,
        "n_labeled": len(labeled_idx_abs),
        "r2": r2,
        "mae": mae,
        "rmse": rmse,
    }

    with open(f"results_{taskname}_{seed}_round{round_id}.json", "w") as f:
        json.dump(results, f)
