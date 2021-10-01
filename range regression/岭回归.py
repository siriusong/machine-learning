import pandas as pd

#1.数据预处理，给每组数据标签
df_names = ['EDUCATION','SOUTH','SEX','EXPERIENCE','UNION','WAGE','AGE','RACE','OCCUPATION','SECTOR','MARR']
df = pd.read_csv('./CPS_85_Wages.txt',sep='\t',names=df_names)
df.shape
df.head()

#2.查看散点图矩阵
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
scatter_matrix(df,figsize=(20,16))
plt.show()

# 3.查看相关系数矩阵
import numpy as np
correlations = df.corr()
correlations

#4.相关系数矩阵可视化
fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(111)
cax = ax.matshow(correlations, vmin=-1, vmax=1)
fig.colorbar(cax)
ticks = np.arange(0,df.shape[1],1)
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_xticklabels(df.columns,rotation=90)
ax.set_yticklabels(df.columns)
plt.show()


########################################################################
#####上面的热力图太丑了 ，我自己写一个
# 基础绘图库
import matplotlib.pyplot as plt
import seaborn as sns
# %matplotlib inline
# 各种细节配置如 文字大小，图例文字等杂项
large = 22; med = 16; small = 12
params = {'axes.titlesize': large,
          'legend.fontsize': med,
          'figure.figsize': (16, 10),
          'axes.labelsize': med,
          'axes.titlesize': med,
          'xtick.labelsize': med,
          'ytick.labelsize': med,
          'figure.titlesize': large}
plt.rcParams.update(params)
plt.style.use('seaborn-whitegrid')
sns.set_style("white")
plt.rc('font', **{'family': 'Microsoft YaHei, SimHei'})  # 设置中文字体的支持
# sns.set(font='SimHei')  # 解决Seaborn中文显示问题，但会自动添加背景灰色网格
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题

# ===================== 热力图 ====================
def heatmap(data, method='pearson', camp='RdYlGn', figsize=(10 ,8), ax=None):
    """
    data: 整份数据
    method：默认为 pearson 系数
    camp：默认为：RdYlGn-红黄蓝；YlGnBu-黄绿蓝；Blues/Greens 也是不错的选择
    figsize: 默认为 10，8
    """
    ## 消除斜对角颜色重复的色块
    #     mask = np.zeros_like(df2.corr())
    #     mask[np.tril_indices_from(mask)] = True
    plt.figure(figsize=figsize, dpi= 80)
    sns.heatmap(data.corr(method=method), \
                xticklabels=data.corr(method=method).columns, \
                yticklabels=data.corr(method=method).columns, cmap=camp, \
                center=0, annot=True, ax=ax)
    # 要想实现只是留下对角线一半的效果，括号内的参数可以加上 mask=mask
######################################################################


#5.划分自变量与因变量
X = df.loc[:,df.columns != 'WAGE']
y = df['WAGE']
X.sample(6)

#6.划分训练集与测试集
from sklearn.model_selection import train_test_split
X_tr,X_ts,y_tr,y_ts = train_test_split(X,y,test_size=0.2)
len(y_tr),len(y_ts)

#7.使用VIF检查方差膨胀因子
from statsmodels.stats.outliers_influence import variance_inflation_factor
vif = pd.DataFrame()
vif["features"] = X.columns
vif["VIF Factor"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif

#8.使用岭回归建立模型
from sklearn.linear_model import Ridge
ridgereg = Ridge(alpha=0.001,normalize=True)
ridgereg.fit(X_tr,y_tr)

#9.查看预测结果前5项
y_ts_pred = ridgereg.predict(X_ts)
y_ts_pred[:5]

#10.查看特征值系数
coef = pd.DataFrame()
coef["features"] = X.columns
coef["coef"] = ridgereg.coef_
coef.append({'features':'INTERCEPTION','coef':ridgereg.intercept_},ignore_index=True)

#11.计算RMSE
rmse_ts = np.sqrt(np.mean(np.square(y_ts_pred-y_ts)))
rmse_ts

#12.使用10折交叉验证计算RMSE
from sklearn.model_selection import cross_val_score
scores = cross_val_score(ridgereg,X,y,cv=10,scoring='neg_mean_squared_error')
rmse_cv = np.sqrt(-scores.mean())
rmse_cv