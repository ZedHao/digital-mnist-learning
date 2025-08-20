import learn as learn
import pandas as pd
import numpy as np
import time
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from data_manager import *
from joblib import dump, load
import pdb

# read_data()
X_train_small, y_train_small, X_test = read_data()

# LR
# begin time
start = int(time.time())
# progressing
"""
lbfgs + l2
"""
# parameters = {'penalty': ['l2'], 'C': [2e-2, 4e-2, 8e-2, 12e-2, 2e-1]}
# lr_clf = LogisticRegression(
#     penalty='l2', solver='lbfgs', multi_class='multinomial', max_iter=800,  C=0.2)





#parameters = {'penalty': ['l1'], 'C': [2e0, 2e1, 2e2]}
#lr_clf= LogisticRegression(penalty='l1', multi_class='ovr', max_iter=800, solver='liblinear' )
'''
网格搜索需要测试 5 组参数：C 2e-2=0.02、0.04、0.08、0.12、0.2（penalty 固定为 'l2'）。
每组参数都要通过5 折交叉验证评估性能（将 1000 个样本分成 5 份，每份 200 个）。
每个参数组合下的模型训练（逻辑回归）最多迭代 800 次（max_iter=800）。、
1. penalty 惩罚项 l2 l1是啥
    
2. 正则化强度 λ=1/0.02=50
    
1.  初始化模型参数
针对 C=0.02（正则化强度 λ=1/0.02=50，强正则化），初始化逻辑回归模型：
权重 w（10 类 ×784 像素 = 7840 个参数）：初始值为 0
偏置 b（10 个参数）：初始值为 0

'''
parameters = {'penalty': ['l2'], 'C': [2e-2, 4e-2, 8e-2, 12e-2, 2e-1]}
lr_clf = LogisticRegression(
   penalty='l2', solver='lbfgs', multi_class='multinomial', max_iter=800,  C=0.2)
gs_clf = GridSearchCV(lr_clf, parameters, n_jobs=1, verbose=True)

gs_clf.fit(X_train_small.astype('float')/256, y_train_small)


# end time
elapsed = (int(time.time()) - start)



if __name__ == '__main__':
    # 最好的参数
    print("best_params_:---", gs_clf.best_params_)
    # 最高的得分
    print("best_score_:---",gs_clf.best_score_)
    # 打印最好的模型
    print("gs_clf.best_estimator_", gs_clf.best_estimator_)
    # 打印各项参数
    print_grid_mean(gs_clf.cv_results_)
    print("Time used:", elapsed)
   # 保存模型
    dump(gs_clf, 'saved_model/lr.pkl')
    #load model
