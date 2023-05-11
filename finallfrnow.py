import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pickle
import random
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

dir='G:\\Documents\\Important Docs\\NMIT\\Final Year Project\\Disease Detection in Plants\\Plant_Dir'
data=[]
cat=['Potato','Tomato']
cat1=['p-earlyblight']
cate='Potato'

if cate=='Potato':
    path=os.path.join(dir,cate)
    for category in cat1:
        path1=os.path.join(path,category)
        label=cat1.index(category)
        for img in os.listdir(path1):
            impath=os.path.join(path1,img)
            plantimg=cv2.imread(impath,1)
            try:
                plantimg=cv2.resize(plantimg,(50,50))
                image=np.array(plantimg).flatten()
                data.append([image,label])
            except Exception as e:
                pass
        
print('Length of data=',len(data))
pick_in=open('sample.pickle','wb')
pickle.dump(data,pick_in)
pick_in.close()
pick_in=open('sample.pickle','rb')
data=pickle.load(pick_in)
pick_in.close()

random.shuffle(data)
features=[]
labels=[]

for feature,label in data:
    features.append(feature)
    labels.append(label)
    
xtrain,xtest,ytrain,ytest=train_test_split(features,labels,test_size=0.25)
model=SVC(C=1,gamma='auto',kernel='poly')
model.fit(xtrain,ytrain)
pick=open('model.sav','wb')
pickle.dump(model,pick)
pick.close()
pick=open('model.sav','rb')
model=pickle.load(pick)
pick.close()

prediction=model.predict(xtest)
accuracy=model.score(xtest,ytest)
print('Accuracy=',accuracy)
print('Prediction',cat1[prediction[0]])
np.array([0])
Mypet=xtest[0].reshape(75,100)
plt.imshow(Mypet,cmap='gray')
plt.show()
