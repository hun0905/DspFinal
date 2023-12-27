#%%
import os
import cv2
from sklearn import svm
import numpy as np
from HOG import getTextureFeature
from Color import getColorFeature
from Shape import getShapeFeatures
from sklearn.metrics import confusion_matrix, accuracy_score
from plot_cm import plot_confusion_matrix
from sklearn.preprocessing import StandardScaler
from pickle import dump
#%% 將訓練用的image存入imgs，label放至Y
train_path = 'training set folder path'
food_class=os.listdir(train_path)
#獲取各個種類食物的資料夾的路徑
food_class_paths = list(map(lambda x:os.path.join(train_path,x),food_class))
Y=[]
X=[]
for path in food_class_paths:
    img_paths=os.listdir(path)#從食物種類資料夾中獲取各個食物影像的路徑
    for img_path in img_paths:
        try:
            img=cv2.imread(os.path.join(path,img_path), cv2.IMREAD_COLOR)
            img=cv2.resize(img, (512, 512))#將食物影像改成統一的大小
            #獲取影像的color, texture, shape的descriptor
            X.append(np.concatenate((getColorFeature(img), getTextureFeature(cv2.resize(img, (64, 64))),getShapeFeatures(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))), axis=0)) 
            #獲取各個影像所對應的label
            Y.append(path.split('/')[-1])
        except:
            continue

#%%保存訓練用的參數和label
dump(X, open('X_train.pkl', 'wb'))
dump(Y, open('Y_train.pkl', 'wb'))


#%%訓練模型並計算其accuracu與confusion matrix

#先將資料進行標準化
scaler = StandardScaler()
X_normal=scaler.fit_transform(X)

#使用support vector machine作為model進行訓練
clf=svm.SVC(kernel='linear',C=2.62,gamma=5.38,probability=True)
# clf= DecisionTreeClassifier(random_state=0)
clf.fit(X_normal,Y)
classNames = clf.classes_

#計算影像在訓練集中的accuracy
y_predict = clf.predict(X_normal)
accuracy = accuracy_score(y_pred=y_predict,y_true=Y)
print(f'accuracy: {accuracy}')

#繪製confusion matrix和印出各類別所對應的precision
cm = confusion_matrix(y_true=Y,y_pred=y_predict)
plot_confusion_matrix(cm , classNames)
cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
print(dict(zip(classNames, cm.diagonal())))

#%% load test set and extract features from test set
#大致同訓練集的步驟
test_path = 'testing dataset path'

test_class=os.listdir(test_path)
food_class_path = list(map(lambda x:os.path.join(test_path,x),test_class))
Y_test=[]
X_test=[]
for path in food_class_path:
    img_paths=os.listdir(path)
    for img_path in img_paths:
        try:
            img=cv2.imread(os.path.join(path,img_path), cv2.IMREAD_COLOR)
            img=cv2.resize(img, (512, 512))
            X_test.append(np.concatenate((getColorFeature(img), getTextureFeature(cv2.resize(img, (64, 64))),getShapeFeatures(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))), axis=0))
            # X_test.append(getColorFeature2(img))
            Y_test.append(path.split('/')[-1])
        except:
            continue
#%%儲存測試集的參數和label
dump(X_test, open('X_test.pk2', 'wb'))
dump(Y_test, open('Y_test.pk2', 'wb'))    
#%% predict accuracy,confusion matrix and classwise precision

#大致同訓練集的步驟
X_test_normal = scaler.transform(X_test)
y_predict_test=clf.predict(X_test_normal)
print(y_predict_test)
accuracy = accuracy_score(y_pred=y_predict_test,y_true=Y_test)
print(f'accuracy: {accuracy}')
cm = confusion_matrix(y_true=Y_test,y_pred=y_predict_test)
plot_confusion_matrix(cm , classNames)
cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
print(dict(zip(classNames, cm.diagonal())))
#%%將訓練完的模型進行保存
# save the model
dump(clf, open('FoodClassifier6.pkl', 'wb'))
# save the scaler
dump(scaler, open('scaler6.pkl', 'wb'))
food_class