
# Sklearn 介绍
https://mp.weixin.qq.com/s?__biz=MzU0MDQ1NjAzNg==&mid=2247559334&idx=1&sn=ea98c833bdb665df9a89d57be3873493&chksm=fb3b4dadcc4cc4bbbd26d6fbeb706a747059203f8431927a8c1ab43fe94650ccd4e51d1669c3&scene=27
1. 分类、回归、聚类、降维、模型选择和预处理
2. 机器学习四要素：数据、任务、性能度量和模型
3. 包括估计器、预测器和转换器。
4. 假设用性能度量 P 来评估机器在某类任务 T 的性能，若该机器通利用经验 E 在任务 T 中改善其性能 P，那么可以说机器对经验 E 进行了学习。
5. performance=quality-metric/task/experience=data/model
6. 数据
   1. 结构化数据和非结构化数据【CNN RNN 卷积神经网络和循环神经网络】/原始数据-加工数据/样本内外数据
   2. 示例 (instance)/特征 (feature) 或输入 (input)/为特征值 (feature value)/标签 (label) 或输出 (output)
   3. /样例 (example)= (特征, 标签)/学习 (learning) 或训练 (training)/训练样例 (training example)，整个集合称为训练集 (training set)
7. task
   1. 有监督学习 (有标签)/无监督学习 (无标签)/半监督学习 (有部分标签)/增强学习 (有评级标签)
   2. Sklearn 里面模型主要处理「有监督学习」和「无监督学习」两类
   3. 分类和回归
      1. 比赛结果赢或输，此类学习任务称为分类 (classification)。
      2. 詹姆斯效率 65.1, 70.3 等等，此类学习任务称为回归 (regression)。
   4. ![img_1.png](img_1.png)
   5. 降维/独热编码
8. 性能度量performance
   1. 回归/分类
      1. 误差函数：预测值和真实值的方差
      2. 错误率/查准率和查全率
   2. ![img_2.png](img_2.png)
   3. ![img_3.png](img_3.png)
9. 核心api
   1. 估计器 (estimator) 拟合器
      1. 有监督学习 model.fit( X_train, y_train )
      2. 无监督学习 model.fit( X_train, y_train )
      3. 拟合之后可以访问 model 里学到的参数，比如线性回归里的特征前的系数 coef_，或 K 均值里聚类标签 labels_。
   2. 预测器 (predictor) 是具有预测功能的估计器 
      1. 转换器也是一种估计器，两者都带拟合功能，估计器里 fit + predict，转换器里 fit + transform。 
         1. 有监督学习的对率回归 预测的类别predict() ，想知道预测该类别的信心predict_proba()。
         2. 无监督学习的K均值
   3. 转换器 (transformer) 是具有转换功能的估计器
      1. 将分类型变量 (categorical) 编码成数值型变量 (numerical) 字符转数字/独热编码
      2. 规范化 (normalize) 或标准化 (standardize) 数值型变量
      3. 警示：fit() 函数只能作用在训练集上，千万不要作用在测试集上，要不然你就犯了数据窥探的错误了！拿标准化举例，用训练集 fit 出来的均值和标准差参数，来对测试集做标准化。
      4. 特征缩放
         1. ![img_4.png](img_4.png)
10. 高级API
    1. 
![img.png](img.png)
## GridSearchCV
1. 人工选择的参数叫超参数，凭经验调或者选不同的参数带入模型选择表现最好的参数
2. 超参数选择不恰当，就会出现欠拟合或者过拟合的问题。
3. 他要求遍历所有可能参数的组合，在面对大数据集和多参数的情况下，非常耗时。
   1. 网格搜索适用于三四个（或者更少）的超参数（当超参数的数量增长时，网格搜索的计算复杂度会呈现指数增长，这时候则使用随机搜索），
## LR 算法 LogisticRegression
 