import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, Matern, ConstantKernel as C
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import numpy as np
from xgboost import XGBRegressor
from matplotlib import pyplot as plt
import warnings
from src.utils import generation_pool, func, initialize, data_extraction


warnings.filterwarnings("ignore")
# Load the dataset

# data = pd.read_csv('data/pan_test.csv')  # Replace with your dataset path
# define seed for reproducibility

target_variable = ['wca', 'q', 'sigma']  # Replace with your target variable name
# Separate features and target variable
# X_test = data.drop(columns=target_variable)  # Replace 'target' with your target variable name
# y_test = data[target_variable]
# test_size = 0.2

train_sizes = [5,7,9,11,13,15,17,19,21,23,25,27,29,31,33,35,37,39,41,43]

GP_matern_mse_train_size_list = []
GP_matern_r2_train_size_list = []
GP_rbf_mse_train_size_list = []
GP_rbf_r2_train_size_list = []
RF_mse_train_size_list = []
RF_r2_train_size_list = []
XGB_mse_train_size_list = []
XGB_r2_train_size_list = []

for train_size in train_sizes:
    print(f'Training set size: {train_size}')
    GP_matern_mse_list = []
    GP_matern_r2_list = []
    GP_rbf_mse_list = []
    GP_rbf_r2_list = []
    RF_mse_list = []
    RF_r2_list = []
    XGB_mse_list = []
    XGB_r2_list = []
    for seed in [40, 41, 42, 43, 44, 45, 46, 47, 48, 49]:
        np.random.seed(seed)

        X_test, X_pool_filtered = generation_pool(seed=seed)
        X_test = X_test.to_numpy(dtype=float)
        y_test = func(X_test)
        y_test = pd.DataFrame(y_test, columns=["wca", "q", "sigma"])

        X_unlabeled = X_pool_filtered.copy() # 参数空间
        initial_idx = initialize(X_unlabeled, train_size, method="random", random_state=seed)
        X_labeled, X_unlabeled = data_extraction(initial_idx, X_unlabeled)
        X_labeled_np = X_labeled.to_numpy(dtype=float)
        y_labeled = func(X_labeled_np)
        y_labeled = pd.DataFrame(y_labeled, columns=["wca", "q", "sigma"])
        # Split the dataset into training and testing sets 训练集占20%，测试集占80%

        # Standardize the features
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_labeled)
        X_test = scaler.transform(X_test)

        d = X_train.shape[1]  # 你的特征维度

        # Constant * Matern(nu=2.5, ARD) + WhiteKernel
        kernel_Matern = (
            C(1.0, (1e-3, 1e3)) *
            Matern(length_scale=np.ones(d), length_scale_bounds=(1e-2, 1e2), nu=2.5)
            +
            WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-8, 1e0))
        )

        # Constant * RBF(ARD) + WhiteKernel
        kernel_RBF = (
            C(1.0, (1e-3, 1e3)) *
            RBF(length_scale=np.ones(d), length_scale_bounds=(1e-2, 1e2))
            +
            WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-8, 1e0))
        )

        # Create Gaussian Process Regressor models
        gp_matern = GaussianProcessRegressor(kernel=kernel_Matern, n_restarts_optimizer=10, random_state=seed)
        gp_rbf = GaussianProcessRegressor(kernel=kernel_RBF, n_restarts_optimizer=10, random_state=seed)
        RF = RandomForestRegressor(n_estimators=100, random_state=seed)
        XGB = XGBRegressor(n_estimators=100, random_state=seed)
        # Fit the models to the training data
        gp_matern.fit(X_train, y_labeled)
        gp_rbf.fit(X_train, y_labeled)
        RF.fit(X_train, y_labeled)
        XGB.fit(X_train, y_labeled)


        ##################################### Autosklearn
        # import autosklearn.regression
        # automl = autosklearn.regression.AutoSklearnRegressor(time_left_for_this_task=3600, per_run_time_limit=300, random_state=seed)
        # automl.fit(X_train, y_train)

        # Make predictions on the test set
        y_pred_matern = gp_matern.predict(X_test)
        y_pred_rbf = gp_rbf.predict(X_test)
        y_pred_rf = RF.predict(X_test)
        y_pred_xgb = XGB.predict(X_test)
        # y_pred_automl = automl.predict(X_test)
        # Evaluate the models
        mse_matern = mean_squared_error(y_test, y_pred_matern)
        r2_matern = r2_score(y_test, y_pred_matern)
        mse_rbf = mean_squared_error(y_test, y_pred_rbf)
        r2_rbf = r2_score(y_test, y_pred_rbf)
        mse_rf = mean_squared_error(y_test, y_pred_rf)
        r2_rf = r2_score(y_test, y_pred_rf)
        mse_xgb = mean_squared_error(y_test, y_pred_xgb)
        r2_xgb = r2_score(y_test, y_pred_xgb)
        # mse_automl = mean_squared_error(y_test_wca, y_pred_automl)
        # r2_automl = r2_score(y_test_wca, y_pred_automl)

        GP_matern_mse_list.append(mse_matern)
        GP_matern_r2_list.append(r2_matern)
        GP_rbf_mse_list.append(mse_rbf)
        GP_rbf_r2_list.append(r2_rbf)
        RF_mse_list.append(mse_rf)
        RF_r2_list.append(r2_rf)
        XGB_mse_list.append(mse_xgb)
        XGB_r2_list.append(r2_xgb)
    GP_matern_mse_mean = np.mean(GP_matern_mse_list)
    GP_matern_r2_mean = np.mean(GP_matern_r2_list)
    GP_rbf_mse_mean = np.mean(GP_rbf_mse_list)
    GP_rbf_r2_mean = np.mean(GP_rbf_r2_list)
    RF_mse_mean = np.mean(RF_mse_list)
    RF_r2_mean = np.mean(RF_r2_list)
    XGB_mse_mean = np.mean(XGB_mse_list)
    XGB_r2_mean = np.mean(XGB_r2_list)

    GP_matern_mse_train_size_list.append(GP_matern_mse_mean)
    GP_matern_r2_train_size_list.append(GP_matern_r2_mean)
    GP_rbf_mse_train_size_list.append(GP_rbf_mse_mean)
    GP_rbf_r2_train_size_list.append(GP_rbf_r2_mean)
    RF_mse_train_size_list.append(RF_mse_mean)
    RF_r2_train_size_list.append(RF_r2_mean)
    XGB_mse_train_size_list.append(XGB_mse_mean)
    XGB_r2_train_size_list.append(XGB_r2_mean)
plt.figure(figsize=(10, 6))
plt.plot(train_sizes, GP_matern_r2_train_size_list, label='GP Matern R2', marker='o')
plt.plot(train_sizes, GP_rbf_r2_train_size_list, label='GP RBF R2', marker='o')
plt.plot(train_sizes, RF_r2_train_size_list, label='Random Forest R2', marker='o')
plt.plot(train_sizes, XGB_r2_train_size_list, label='XGBoost R2', marker='o')
plt.xlabel('Training Set Size')
plt.ylabel('R² Score')
plt.title('Model Performance with Varying Training Set Sizes')
plt.legend()
# print(f'Matern Kernel - MSE: {mse_matern:.4f}, R²: {r2_matern:.4f}')
# print(f'RBF Kernel - MSE: {mse_rbf:.4f}, R²: {r2_rbf:.4f}') 
# print(f'Random Forest - MSE: {mse_rf:.4f}, R²: {r2_rf:.4f}')
# print(f'XGBoost - MSE: {mse_xgb:.4f}, R²: {r2_xgb:.4f}')
# # print(f'AutoSklearn - MSE: {mse_automl:.4f}, R²: {r2_automl:.4f}')
plt.show()

