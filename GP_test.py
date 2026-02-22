import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, Matern, ConstantKernel as C
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import numpy as np
# Load the dataset
data = pd.read_csv('FEA_output_pos.csv')  # Replace with your dataset path
# define seed for reproducibility
seed = 42
np.random.seed(seed)


# Separate features and target variable
X = data.drop(columns=['max_uz'])  # Replace 'target' with your target variable name
y = data['max_uz']  # Replace 'target' with your target variable name   
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)
# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
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
# Fit the models to the training data
gp_matern.fit(X_train, y_train)
gp_rbf.fit(X_train, y_train)
RF.fit(X_train, y_train)
##################################### Autosklearn
import autosklearn.regression
automl = autosklearn.regression.AutoSklearnRegressor(time_left_for_this_task=3600, per_run_time_limit=300, random_state=seed)
automl.fit(X_train, y_train)

# Make predictions on the test set
y_pred_matern = gp_matern.predict(X_test)
y_pred_rbf = gp_rbf.predict(X_test)
y_pred_rf = RF.predict(X_test)
y_pred_automl = automl.predict(X_test)
# Evaluate the models
mse_matern = mean_squared_error(y_test, y_pred_matern)
r2_matern = r2_score(y_test, y_pred_matern)
mse_rbf = mean_squared_error(y_test, y_pred_rbf)
r2_rbf = r2_score(y_test, y_pred_rbf)
mse_rf = mean_squared_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)
mse_automl = mean_squared_error(y_test, y_pred_automl)
r2_automl = r2_score(y_test, y_pred_automl)

print(f'Matern Kernel - MSE: {mse_matern:.4f}, R²: {r2_matern:.4f}')
print(f'RBF Kernel - MSE: {mse_rbf:.4f}, R²: {r2_rbf:.4f}') 
print(f'Random Forest - MSE: {mse_rf:.4f}, R²: {r2_rf:.4f}')
print(f'AutoSklearn - MSE: {mse_automl:.4f}, R²: {r2_automl:.4f}')


