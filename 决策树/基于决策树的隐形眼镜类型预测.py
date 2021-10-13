import pandas as pd
import matplotlib.pyplot as plt

##加载划分训练集/测试集的函数
from sklearn.model_selection import train_test_split

##加载决策树分类器
from sklearn.tree import DecisionTreeClassifier

##加载交叉验证分数计算
from sklearn.model_selection import cross_val_score

##加载分类报告，混淆矩阵，准确率
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score

##准备数据集的字段名
names=['age_of_the_patient','spectacle_prescription','astigmatic','tear_production_rate','contact_lenses']

# 使用pandas读取csv格式数据
# 数据没有表头，因此header=None
# 数据有序号列，为第0列，因此index_col=0
data = pd.read_csv('./lenses.csv', header=None, index_col=0,names=names)

##对数据集进行可视化

##绘制柱形图，尺寸为10*8
data.hist(figsize=(10,8)),
plt.show()

##划分数据集，所有非contact_lenses均为自变量
X=data.loc[:,data.columns!='contact_lenses']
y=data.loc[:,data.columns=='contact_lenses']

##建立决策树分类模型，用基尼系数
model=DecisionTreeClassifier(criterion='gini').fit(X, y)

##预测结果
pre=model.predict(X)

##查看混淆矩阵
print(confusion_matrix(y,pre))