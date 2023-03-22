
#keras使用RNN预测 中国平安银行数据 预测股市

import matplotlib_inline
import numpy as np
from keras.layers import Dense,SimpleRNN
from keras.models import Sequential
import pandas as pd
from matplotlib import pyplot as plt

#define x,y
def extract_data(data,slide):
    x ,y = [] , []
    for i in range(len(data) - slide):
        x.append([a for a in data[i:i+slide]])
        y.append(data[i+slide])

    x = np.array(x)
    x = x.reshape(x.shape[0],x.shape[1],1)
    return x,y

data = pd.read_csv('D:\AI_coding\DL_NLP_Project\dataset\zgpa_train.csv')
data.head()
# print(data)
price = data.loc[:,'close']
price.head()

#归一化处理
price_nom = price/max(price)
# print(price_nom)

# %matplotlib_inline
fig1 = plt.figure(figsize=(8,5))
plt.plot(price)
plt.title('close price')
plt.xlabel('time')
plt.ylabel('price')
# plt.show()

time_step = 8
x,y = extract_data(price_nom,time_step)
# print(x)

#建立模型
model = Sequential()
model.add(SimpleRNN(units=5,input_shape=(time_step,1),activation='relu'))
model.add(Dense(units = 1,activation='linear'))
model.compile(optimizer='adam',loss='mean_squared_error')
model.summary()

#训练数据
model.fit(x,y,batch_size=30,epochs=200,)

y_train_predict = model.predict(x) *max(price)
y_train = [i * max(price) for i in y]

fig2 = plt.figure(figsize=(8,5))
plt.plot(y_train,label ='real price')
plt.plot(y_train_predict, label = 'prdict price')
plt.title('close price')
plt.xlabel('time')
plt.ylabel('price')
plt.show()