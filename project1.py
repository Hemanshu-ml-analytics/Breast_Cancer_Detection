

print("Breast-Cancer Detection classification problem")


# import libaries

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
import matplotlib.pyplot as plt

# load data
dataset=load_breast_cancer()
print(dataset)

# manage data properly
framed_data=pd.DataFrame(dataset.data,columns=dataset.feature_names)
print(framed_data.head(3))

# create a label based on this data which have already with name target
framed_data['label']=dataset.target
print(framed_data)

# x and y give data
x=framed_data.drop('label',axis=1)
y=framed_data['label']
print(x)
print(y)

# check shape ao info /describe
print(x.shape)
print(y.shape)

# value count for check how many zeros and one or imbalanced data detection
print(framed_data['label'].value_counts())

# check not a nul value
print(framed_data.isnull().sum())

# train test splitting
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
print(np.shape(x_train))
print(np.shape(x_test))
print(np.shape(y_train))

# standard scaller
from sklearn.preprocessing import StandardScaler
scaller=StandardScaler()
std_xtrain=scaller.fit_transform(x_train)
std_xtest=scaller.fit_transform(x_test)

# now create the sequential deep model

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Flatten,Dropout



model=Sequential()
Flatten(input_shape=(30,))

model.add(Dense(20, activation='relu'))
model.add(Dense(2, activation='sigmoid'))

# comile it
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])

# fit data 
history=model.fit(std_xtrain,y_train,epochs=10)
print("model trained succesfully")

# check the accuracy of model
loss,accuracy=model.evaluate(std_xtest,y_test)
print("loss=",loss,"accuracy=",accuracy)

# plot a diagram for epochs vs accuracy
plt.plot(history.history['accuracy'],label='accuracy',color='red')
plt.legend()
plt.title("accuracy vs epochs")
plt.grid(True)
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.show()
         



# plot diagram for epochs vs loss


# prediction on std_xtest[0]
prediction=model.predict(std_xtest)
print("prediction=",prediction[0])

# now using argmax method because it returned the index of highest one element in list
classification=[np.argmax(i) for i in prediction]
print(classification)


# testing on unseen data

input_data = (11.76,21.6,74.72,427.9,0.08637,0.04966,0.01657,0.01115,0.1495,0.05888,0.4062,1.21,2.635,28.47,0.005857,0.009758,0.01168,0.007445,0.02406,0.001769,12.98,25.72,82.98,516.5,0.1085,0.08615,0.05523,0.03715,0.2433,0.06563)

# change the input_data to a numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshape the numpy array as we are predicting for one data point
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

# standardizing the input data
input_data_std = scaller.transform(input_data_reshaped)

prediction = model.predict(input_data_std)
print(prediction)

prediction_label = [np.argmax(prediction)]
print(prediction_label)

if(prediction_label[0] == 0):
  print('The tumor is Malignant')

else:
  print("tumor is benign")