from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn import datasets
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
data=pd.read_table('./datingTestSet2.txt',header=None,names=['var1','var2','var3','label'])
data.sample(5)
x=data.iloc[:,:3]
y=data.label
(trainData,testData,trainLabels,testLabels)=train_test_split(x,y,test_size=0.25,random_state=40)

model=KNeighborsClassifier(n_neighbors=3)
model.fit(trainData,trainLabels)
predictions=model.predict(testData)
print(classification_report(testLabels,predictions))