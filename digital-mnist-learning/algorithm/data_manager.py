import pandas as pd

# read train & test data
def read_data():
    root_path = "../digital_minist_data/"
    dataset = pd.read_csv(root_path + "train.csv")
    X_train = dataset.values[0:, 1:]
    y_train = dataset.values[0:, 0]
    # 1000以下的数据量，遍历时间在10s之内，方便测试
    X_train_small = X_train[:1000, :]
    y_train_small = y_train[:1000]
    X_test = pd.read_csv(root_path +"test.csv").values
    return X_train_small, y_train_small, X_test


# handler grid result
def sorted_grid_scores(gridScores):
    def sort_by_mean(val):
        return val['mean_test_score']


    sorted_scores = sorted(gridScores,
                           key=sort_by_mean,
                           reverse=True)
    return sorted_scores





lrExample= {
    "scoring": None,  # 评分标准，用于评估模型性能的指标。在此处为None，表示未指定特定的评分标准。
    # 使用的模型估计器，这里是一个逻辑回归模型的字符串表示，包括模型的参数设置。
    "estimator": "LogisticRegression(C=4, max_iter=800, multi_class='ovr', penalty='l1', solver='liblinear')",
    "n_jobs": 1,  # 并行执行的作业数，用于加速计算。
    "refit": True,  # 是否在找到最佳参数后重新拟合整个数据集。
    "cv": None,  # 交叉验证的折数，用于模型评估的数据划分。
    "verbose": True,  # 控制输出的详细程度，此处为True，表示输出详细信息。
    "pre_dispatch": "2*n_jobs",  # 预分派作业数的数量，用于控制并行计算的资源分配。
    "error_score": "nan",  # 如果拟合过程中发生错误，用于替代分数的值。
    "return_train_score": False,  # 是否返回训练分数，此处为False，表示不返回训练分数。
    "param_grid": {
        # 待搜索的参数空间，包括正则化惩罚项。
        # 理解这里的l1 l2
       "penalty": ["l1", "l2"],
    "C": [2.0, 20.0, 200.0, 2000.0]  # 正则化强度。
    },
    "multimetric_": False,  # 是否使用多个指标进行评分。
    "best_index_": 1,  # 最佳参数组合的索引。
    "best_score_": 0.8460000000000001,  # 在交叉验证中获得的最佳平均分数。
    "best_params_": {
                        "C": 2.0,  # 最佳参数组合，包括正则化强度。
    "penalty": "l2"  # 包括正则化惩罚项。
    },
    "best_estimator_": "LogisticRegression(C=2.0, max_iter=800, multi_class='ovr', solver='liblinear')",  # 在整个数据集上使用最佳参数组合训练得到的最佳估计器的字符串表示。
    "refit_time_": 0.24620699882507324,  # 重新拟合整个数据集所花费的时间。
    "scorer_": "<sklearn.metrics._scorer._PassthroughScorer object at 0x16be88f40>",  # 评分器对象，用于评估模型性能的对象。

        "cv_results_": {  # 交叉验证的结果，包括平均拟合时间、平均评分时间、参数组合及其得分等信息。
            "mean_fit_time": [0.17564363, 0.19697924, 0.28087463, 0.26051641, 0.23892961, 0.25506101, 0.1367609, 0.24991302],  # 每个参数组合的平均拟合时间
            "std_fit_time": [0.02732746, 0.0048617, 0.02309295, 0.01606831, 0.01841419, 0.01100812, 0.01260539, 0.01559765],  # 每个参数组合的拟合时间标准差
            "mean_score_time": [0.0039722, 0.00088696, 0.00234394, 0.00197196, 0.00101399, 0.00140786, 0.00083237, 0.00218983],  # 每个参数组合的平均评分时间
            "std_score_time": [0.0030307, 0.00033643, 0.00198331, 0.00104432, 0.00061824, 0.00059569, 0.00030145, 0.00188721],  # 每个参数组合的评分时间标准差
            "param_C": [2.0, 2.0, 20.0, 20.0, 200.0, 200.0, 2000.0, 2000.0],  # 参数 C 的取值
            "param_penalty": ["l1", "l2", "l1", "l2", "l1", "l2", "l1", "l2"],  # 参数 penalty 的取值
            "params": [  # 参数组合列表
            {"C": 2.0, "penalty": "l1"},
            {"C": 2.0, "penalty": "l2"},
            {"C": 20.0, "penalty": "l1"},
            {"C": 20.0, "penalty": "l2"},
            {"C": 200.0, "penalty": "l1"},
            {"C": 200.0, "penalty": "l2"},
            {"C": 2000.0, "penalty": "l1"},
            {"C": 2000.0, "penalty": "l2"}
            ],
            "split0_test_score": [0.785, 0.805, 0.78, 0.79, 0.785, 0.77, 0.79, 0.765],  # 第 1 折的测试分数
            "split1_test_score": [0.835, 0.84, 0.815, 0.83, 0.795, 0.825, 0.795, 0.815],  # 第 2 折的测试分数
            "split2_test_score": [0.83, 0.85, 0.83, 0.85, 0.835, 0.835, 0.84, 0.83],  # 第 3 折的测试分数
            "split3_test_score": [0.845, 0.855, 0.805, 0.855, 0.825, 0.85, 0.83, 0.845],  # 第 4 折的测试分数
            "split4_test_score": [0.87, 0.88, 0.87, 0.88, 0.86, 0.87, 0.865, 0.87],  # 第 5 折的测试分数
            # 口径 每组超参数组合在交叉验证中的平均测试分数。
            "mean_test_score": [0.833, 0.846, 0.82, 0.841, 0.82, 0.83, 0.824, 0.825],  # 平均测试分数
            # 每组超参数组合在交叉验证中测试分数的标准差 减去平均 平方开方
            "std_test_score": [0.02767671, 0.02437212, 0.02983287, 0.03006659, 0.02720294, 0.03361547, 0.02817801, 0.03507136],  # 测试分数的标准差
            "rank_test_score": [3, 1, 8, 2, 7, 4, 6, 5]  # 测试分数的排名
            },
            "n_splits_": 5 # 执行交叉验证时使用的折数。
}
def print_grid_mean(gridScores, sorted=True):
    # 可以看下上面的例子
    print("\ngrid_scores_:")
    for k,v  in gridScores.items():
        print(k,"------",v)
    print()


"""
Print grid result

----------
grid_scores_ : list of named tuples

  * ``parameters``, a dict of parameter settings
  * ``mean_validation_score``, the mean score over the cross-validation folds
  * ``cv_validation_scores``, the list of scores for each fold

"""
def print_grid_mean_new(gridScores, sorted=True):
    print("\ngrid_scores_:")
    print("mean score | scores.std() * 2 | params")
    sorted_scores = gridScores
    if sorted:
        sorted_scores = sorted_grid_scores(gridScores)

    for params, mean_score, scores in sorted_scores:
        print("%0.3f      | (+/-%0.03f)       | %r" % (mean_score, scores.std() * 2, params))
    print()