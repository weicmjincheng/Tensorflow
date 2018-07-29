
# _*_coding:utf-8 _*_
"""
@Time    :2018/6/25 11:33
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
# 定义三个占位符
x = tf.placeholder(tf.float32,[None,784])
y = tf.placeholder(tf.float32,[None,10])
keep_prob = tf.placeholder(tf.float32)
# 增加学习率变量
lr = tf.Variable(0.001,dtype=tf.float32)

# 创建神经网络
W1 = tf.Variable(tf.truncated_normal([784,500],stddev=0.1))
b1 = tf.Variable(tf.zeros([500])+0.1)
L1 = tf.nn.tanh(tf.matmul(x,W1)+b1)
L1_drop = tf.nn.dropout(L1,keep_prob)

W2 = tf.Variable(tf.truncated_normal([500,300],stddev=0.1))
b2 = tf.Variable(tf.zeros([300])+0.1)
L2 = tf.nn.tanh(tf.matmul(L1_drop,W2)+b2)
L2_drop = tf.nn.dropout(L2,keep_prob)

W3 = tf.Variable(tf.truncated_normal([300,10],stddev=0.1))
b3 = tf.Variable(tf.zeros([10])+0.1)
prediction = tf.nn.softmax(tf.matmul(L2_drop,W3)+b3)
# 定义二次代价函数
# loss = tf.reduce_mean(tf.square(y-prediction))
loss =tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=prediction))
# 梯度下降法
train_step = tf.train.AdamOptimizer(lr).minimize(loss)
# 初始化变量
init = tf.global_variables_initializer()
# 求准确率    求最大的值是在那个位置
score = tf.equal(tf.argmax(y,1),tf.argmax(prediction,1))
# 将true转换成1将false转换成0
accuracy = tf.reduce_mean(tf.cast(score,tf.float32))
with tf.Session() as sess:
    sess.run(init)
    for epoch in range(51):
        sess.run(tf.assign(lr,0.001*(0.95**epoch)))
        for batch in range(m_batch):
            batch_xs,batch_ys = mnist.train.next_batch(batch_size)
            sess.run(train_step,feed_dict={x:batch_xs,y:batch_ys,keep_prob:1.0})
        learing_rate = sess.run(lr)
        acc = sess.run(accuracy,feed_dict={x:mnist.test.images,y:mnist.test.labels,keep_prob:1.0 })
        print("当前周期是："+str(epoch)+"  测试准确率："+str(acc)+"  学习率："+str(learing_rate))



"""
当前周期是：0  测试准确率：0.9547  学习率：0.001
当前周期是：1  测试准确率：0.9604  学习率：0.00095
当前周期是：2  测试准确率：0.9688  学习率：0.0009025
当前周期是：3  测试准确率：0.9687  学习率：0.000857375
当前周期是：4  测试准确率：0.9723  学习率：0.00081450626
当前周期是：5  测试准确率：0.9755  学习率：0.0007737809
当前周期是：6  测试准确率：0.9766  学习率：0.0007350919
当前周期是：7  测试准确率：0.9756  学习率：0.0006983373
当前周期是：8  测试准确率：0.9768  学习率：0.0006634204
当前周期是：9  测试准确率：0.9779  学习率：0.0006302494
当前周期是：10  测试准确率：0.9789  学习率：0.0005987369
当前周期是：11  测试准确率：0.9771  学习率：0.0005688001
当前周期是：12  测试准确率：0.9792  学习率：0.0005403601
当前周期是：13  测试准确率：0.9766  学习率：0.0005133421
当前周期是：14  测试准确率：0.9811  学习率：0.000487675
当前周期是：15  测试准确率：0.9805  学习率：0.00046329122
当前周期是：16  测试准确率：0.9805  学习率：0.00044012666
当前周期是：17  测试准确率：0.9806  学习率：0.00041812033
当前周期是：18  测试准确率：0.9805  学习率：0.00039721432
当前周期是：19  测试准确率：0.9815  学习率：0.0003773536
当前周期是：20  测试准确率：0.9799  学习率：0.00035848594
"""