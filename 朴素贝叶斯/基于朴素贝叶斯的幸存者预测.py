'''
数据集包含泰坦尼克号891名乘员的基本信息，及幸存数据。字段说明如下：

survival: Survival 0 = No, 1 = Yes

pclass: Ticket class 1 = 1st, 2 = 2nd, 3 = 3rd
sex: Sex
Age: Age in years
sibsp: # of siblings / spouses aboard the Titanic
parch: # of parents / children aboard the Titanic
ticket: Ticket number
fare: Passenger fare
cabin: Cabin number
embarked: C = Cherbourg, Q = Queenstown, S = Southampton
'''
##读取数据
import pandas as pd
data=pd.read_csv("./titanic.csv")
data.head()

##查看数据的统计性描述
data.describe()

##查看缺失值
data.isnull().sum()

##查看散点图矩阵
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
scatter_matrix(data.iloc[:,1:],figsize=(12,10))
plt.show()

##离散变量转换成独热编码
import numpy as np
data['Sex_cleaned']=np.where(data["Sex"]=="male",0,1)
data["Embarked_cleaned"]=np.where(data["Embarked"]=="S",0,
np.where(data["Embarked"]=="C",1,
 np.where(data["Embarked"]=="Q",2,3)))
col_name = ["Survived","Pclass","Sex_cleaned","Age","SibSp","Parch","Fare","Embarked_cleaned"]

##去除缺失值
data=data[col_name].dropna(axis=0,how='any')
data.head()

##划分自变量和因变量
X = data.loc[:,data.columns!='Survived']
y = data.loc[:,data.columns=='Survived']

##划分训练集和测试集
from sklearn.model_selection import train_test_split
X_tr,X_ts,y_tr,y_ts = train_test_split(X,y,random_state=2)
X_tr.shape,X_ts.shape

##建立朴素贝叶斯模型
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(X_tr,y_tr.values.ravel())

##对测试集进行预测 选择f1值
y_pred = gnb.predict(X_ts)
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_val_score
f1_score(y_ts,y_pred)

##计算5折交叉验证f1-score
scores = cross_val_score(gnb,X,y.values.ravel(),cv=5,scoring='f1')
print('5 fold cross validation f1-score: %.4f'%scores.mean())

##绘制ROC曲线，并计算概率阈值
y_tr_pred_prob = gnb.predict_proba(X_tr)
y_tr_pred_prob = [y for x,y in y_tr_pred_prob]
from sklearn.metrics import roc_curve,roc_auc_score
fpr, tpr, thresh = roc_curve(y_tr, y_tr_pred_prob)
df = pd.DataFrame(dict(fpr=fpr, tpr=tpr))
idx = np.argmax(tpr-fpr)
Thresh = thresh[idx]
myAUC = roc_auc_score(y_tr, y_tr_pred_prob)
plt.figure(figsize=(8,6))
plt.plot(fpr,tpr)
plt.scatter(fpr[idx],tpr[idx],c='r')
plt.text(fpr[idx],tpr[idx],'(FPR %.4f,\n TPR %.4f,\n Thresh %.4f,\n AUC %.4f)'%(fpr[idx],tpr[idx],Thresh,myAUC),va='top')
plt.title('ROC curve')
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.grid(True)

##使用阈值重新预测存活
y_ts_pred_prob = gnb.predict_proba(X_ts)
y_ts_pred = (y_ts_pred_prob[:,1]>Thresh) *1
y_ts_pred

##计算测试集ROC阈值预测结果f1-score
f1_score(y_ts,y_ts_pred)

'''
本试验中，测试集f1-score得分0.7368。

5折交叉验证计算f1-score得分0.7255。

ROC曲线下面积，AUC为0.8322。

ROC选取概率阈值0.5415。

根据概率阈值计算出的测试集预测值，f1_score得分0.7328。

使用ROC阈值计算概率阈值，和默认的朴素贝叶斯模型，效果差距不大。
'''