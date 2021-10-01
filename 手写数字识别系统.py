
###读取数据
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn import datasets
import numpy as np
from sklearn.model_selection import train_test_split
mnist= datasets.load_digits()


(trainData,testData,trainLabels,testLabels)=train_test_split(np.array(mnist.data),mnist.target,test_size=0.25,random_state=42)
import matplotlib.pyplot as plt
for i in range(9):
    plt.subplot(3,3,i+1)
    plt.imshow(trainData[i].reshape(8,8))
    
    
model=KNeighborsClassifier(n_neighbors=3)
model.fit(trainData,trainLabels)
predictions=model.predict(testData)
print(classification_report(testLabels,predictions))