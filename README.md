SIM-PAN 和 SIM-TRC 的 参数范围在论文中有

SIM PAN参数范围：
![alt text](image.png)
SIM TRC参数范围
![alt text](image-1.png)

generate_json.py
用于生成训练用的参数空间的json

run_all_methods.py
使用所有方法预测下一轮特征值

export_round1_candidates.py
取生成的特征值导出为训练仿真用的csv文件


GP+US: Performance of uncertainty-based active learning for efficient approximation of black-box functions in materials science
GP+US: Benchmarking the acceleration of materials discovery by sequential learning
GP+US: Adaptive sampling assisted surrogate modeling of initial failure envelopes of composite structures
GP+US(BO for 最大预测目标): Machine-learning-assisted development and theoretical consideration for the Al2Fe3Si3 thermoelectric material
GP+US(BO for 最大预测目标): Designing nanostructures for phonon transport via bayesian optimization.


SVM（RBF核）+ GS/IGS: Exploring active learning strategies for predictive models in mechanics of materials
SVR + Enhanced Query-by-Committee : Applying enhanced active learning to predict formation energy
RF + XGB + QBC（Query-by-Committee）: Exploiting redundancy in large materials datasets for efficient machine learning with less data
NN + Query-by-Committee: Active learning and element-embedding approach in neural networks for infinite-layer versus perovskite oxides

GBR+ Query-by-Committee(SVR（Support Vector Regression）+ GBR（Gradient Boosting Regression）+ FR（Random Forest Regression）+ ABR（AdaBoost Regression）+ KRR（Kernel Ridge Regression）): Active learning for the power factor prediction in diamond-like thermoelectric materials

接下来的工作：
0. 确定所有实验的参数范围。
1. 三个仿真分别生成一个示例性质的数据集，使用拉丁超立方scipy.stats.qmc.LatinHypercube进行采样。200个数据点
2. 对照试验设计：