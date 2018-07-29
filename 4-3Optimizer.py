# _*_coding:utf-8 _*_
"""
@Time    :2018/6/25 11:15
@Author  :weicm
#@Software: PyCharm
"""


import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# 需要载入数据    one_hot编码
mnist = input_data.read_data_sets("MNIST.data",one_hot=True)
batch_size = 100
# 计算一共有多少个批次
m_batch = mnist.train.num_examples // batch_size
# 定义两个占位符
x = tf.placeholder(tf.float32,[None,784])
y = tf.placeholder(tf.float32,[None,10])
# 创建神经网络
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))
prediction = tf.nn.softmax(tf.matmul(x,W)+b)
# 定义二次代价函数
# loss = tf.reduce_mean(tf.square(y-prediction))
loss =tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=prediction))

# 梯度下降法
# train_step = tf.train.GradientDescentOptimizer(0.2).minimize(loss)
# 使用新的优化器
train_step = tf.train.AdamOptimizer(1e-2).minimize(loss)
# 初始化变量
init = tf.global_variables_initializer()
# 求准确率    求最大的值是在那个位置
score = tf.equal(tf.argmax(y,1),tf.argmax(prediction,1))
# 将true转换成1将false转换成0
accuracy = tf.reduce_mean(tf.cast(score,tf.float32))
with tf.Session() as sess:
    sess.run(init)
    for epoch in range(21):
        for batch in range(m_batch):
            batch_xs,batch_ys = mnist.train.next_batch(batch_size)
            sess.run(train_step,feed_dict={x:batch_xs,y:batch_ys})

        acc = sess.run(accuracy,feed_dict={x:mnist.test.images,y:mnist.test.labels})
        print("当前周期是："+str(epoch)+"  准确率："+str(acc))