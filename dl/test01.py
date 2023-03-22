
#模拟LSTM内部两神经元的交互

import numpy as np

def rand_arr(a,b,*args):
    np.random.seed(0)
    return np.random.rand(*args)*(b-a) + a


def sigmod(x):
    return 1./(1+np.exp(-x))


class LSTM:
    def __init__(self,mem_cell_ct,x_dim):
        self.mem_cell_ct = mem_cell_ct
        self.x_dim = x_dim
        concat_len = x_dim + mem_cell_ct

        self.wg = rand_arr(-0.1,0.1,mem_cell_ct,concat_len)
        self.wi = rand_arr(-0.1,0.1,mem_cell_ct,concat_len)
        self.wf = rand_arr(-0.1,0.1,mem_cell_ct,concat_len)
        self.wo = rand_arr(-0.1,0.1,mem_cell_ct,concat_len)

        self.bg = rand_arr(-0.1,0.1,mem_cell_ct)
        self.bi = rand_arr(-0.1,0.1,mem_cell_ct)
        self.bf = rand_arr(-0.1,0.1,mem_cell_ct)
        self.bo = rand_arr(-0.1,0.1,mem_cell_ct)

        self.s = np.zeros(mem_cell_ct)
        self.h = np.zeros(mem_cell_ct)

        self.x = None
        self.xc = None

    def fun_lstm(self,x,s_prev=np.zeros_like(np.zeros(5)),h_prev=np.zeros_like(np.zeros(5))):
        # if s_prev == None:s_prev = np.zeros_like(self.s)
        # if h_prev == None:h_prev = np.zeros_like(self.h)
        self.s_prev = s_prev
        self.h_prev = h_prev
        
        xc = np.hstack((x,h_prev))
        self.g = np.tanh(np.dot(self.wg,xc) + self.bg)
        self.i = sigmod(np.dot(self.wi,xc) + self.bi)
        self.f = sigmod(np.dot(self.wf,xc) + self.bf)
        self.o = sigmod(np.dot(self.wo,xc) + self.bo)

        self.s = self.g * self.i +s_prev * self.f
        self.h = self.s *self.o
        self.x = x
        self.xc = xc


if __name__ == '__main__':
    np.random.seed(0)
    mem_cell_ct = 5
    x_dim = 4
    concat_len = x_dim + mem_cell_ct
    ltsm1 = LSTM(mem_cell_ct,x_dim)
    ltsm2 = LSTM(mem_cell_ct,x_dim)
    y_list = [-0.5,0,2,0.1,-0.5]
    input_val_arr = [np.random.random(x_dim) for _ in y_list]

    # print(input_val_arr[0])
    ltsm1.fun_lstm(input_val_arr[0])
    ltsm2.fun_lstm(input_val_arr[0],ltsm1.s,ltsm1.h)
    print('ltsm1的参数')
    print(ltsm1.s_prev)
    print(ltsm1.h_prev)
    print(ltsm1.f)
    print(ltsm1.i)
    print(ltsm1.o)
    print(ltsm1.x)
    print(ltsm1.h)
    print(ltsm1.s)

    print('ltsm2的参数')
    print(ltsm2.s_prev)
    print(ltsm2.h_prev)
    print(ltsm2.f)
    print(ltsm2.i)
    print(ltsm2.o)
    print(ltsm2.x)
    print(ltsm2.h)
    print(ltsm2.s)
