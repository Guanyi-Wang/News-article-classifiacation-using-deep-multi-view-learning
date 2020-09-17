# from sklearn.datasets import fetch_rcv1
#
# train = fetch_rcv1(subset='train')
# test = fetch_rcv1(subset='test')
import numpy as np
a = np.random.rand(50,31,1,2)
b = np.random.rand(50,30,1,2)
c = np.random.rand(50,29,1,2)
a = np.reshape(a,[-1,62])
b = np.reshape(b,[-1,60])
x = []
x.append(a)
x.append(b)
x = np.concatenate(x, 1)
print(np.shape(x))