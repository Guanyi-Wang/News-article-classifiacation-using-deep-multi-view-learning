from sklearn .datasets import fetch_rcv1
train = fetch_rcv1(subset='train')
test = fetch_rcv1(subset='test')

x_train = train.data.data
x_test = test.data.data

y_train = train.target.data
y_test = train.target.data
print()