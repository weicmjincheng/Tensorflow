# _*_coding:utf-8 _*_
"""
@Time    :2018/6/24 17:09
@Author  :weicm
#@Software: PyCharm
"""
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# 使用numpy生成两百个均匀分布随机点
x_data = np.linspace(-0.8,0.8,200)[:,np.newaxis]
# print(x_data)
noise = np.random.normal(0,0.02,x_data.shape)
y_data = np.square(x_data)+noise
# 定义两个占位符
x = tf.placeholder(tf.float32,[None,1])
y = tf.placeholder(tf.float32,[None,1])
# 定义神经网络中间层
Weight_L1 = tf.Variable(tf.random_normal([1,10]))
biases_L1 = tf.Variable(tf.zeros([1,10]))
Wx_plus_b_L1 = tf.matmul(x,Weight_L1) + biases_L1
L1 = tf.nn.tanh(Wx_plus_b_L1)
# 定义输出层
Weight_L2 = tf.Variable(tf.random_normal([10,1]))
biases_L2 = tf.Variable(tf.zeros([1,1]))
Wx_plus_b_L2 = tf.matmul(L1,Weight_L2) + biases_L2
prediction = tf.nn.tanh(Wx_plus_b_L2)
# 定义代价函数
loss = tf.reduce_mean(tf.square(y-prediction))
# 使用梯度下降法
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for _ in range(2000):
        sess.run(train_step,feed_dict={x:x_data,y:y_data})

    # 获得预测值
    prediction_value = sess.run(prediction,feed_dict={x:x_data})
    # 绘图
    plt.figure()
    plt.scatter(x_data,y_data)
    plt.plot(x_data,prediction_value,'r-',lw=5)
    plt.show()
