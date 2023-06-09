import numpy as np

class Sigmoid:
    def __init__(self):
        self.params = []

    def forward(self,x):
        return 1/(1+np.exp(-x))

class Affine:
    def __init__(self,W,b):
        self.params = [W,b]

    def foeward(self,x):
        W,b = self.params
        out = np.dot(x,W) +b
        return out

class TwoLayerNet:
    def __init__(self,input_size,hidden_size,output_size):
        I,H,O = input_size,hidden_size,output_size

        #初始化权重和偏置
        W1 = np.random.randn(I,H)
        W2 = np.random.randn(H,O)
        b1 = np.random.randn(H)
        b2 = np.random.randn(O)

        #添加层
        self.layers = {
            Affine(W1,b1),
            Sigmoid(),
            Affine(W2,b2)
        }

        #将所有参数权重保存到列表中

        self.params =[]
        for layer in self.layers:
            self.params += layer.params

    def predict(self,x):
        for layer in self.layers:
            x = layer.foeward(x)
        return x
