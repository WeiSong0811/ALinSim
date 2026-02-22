SIM-PAN 和 SIM-TRC 的 参数范围在论文中有



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

    - Finite-Element-Analysis-of-Concrete-using-Python
        - 输入：
            - thk板厚 (100~250) 100 110 120 130 140 150 160 170 180 190 200 210 220 230 240 250 16个
            - 板长/板宽:[1.0 1.25 1.5 1.75 2.0 2.25] 6个
            - 板宽: [1000 1500 2000 2500 3000 3500 4000 4500 5000] 9个
            - E: 24000 26700 30100 32800 34800 37400 39600 42200 来源于Table 3.1.2  8个
            - ρ：2400 kg/m3 固定
            - LL: -1.5 -2.0 -2.5 -3.0 -3.5 -4.0 -5.0 7个
            - SDL -0 -0.25 -1.0 -1.25 -1.5 5个
        - 输出：
            - max|Uz|

    - SIM PAN参数范围：注意，参数是连续的
        ![alt text](image.png)
    - SIM TRC参数范围：注意，参数是离散的
        ![alt text](image-1.png)

1. 三个仿真分别生成一个示例性质的数据集，使用拉丁超立方scipy.stats.qmc.LatinHypercube进行采样。200个数据点。

2. 对照试验设计：