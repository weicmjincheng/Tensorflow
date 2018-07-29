# _*_coding:utf-8 _*_
"""
@Time    :2018/6/25 10:14
@Author  :weicm
#@Software: PyCharm
效果：模型训练的时候需要dropout，当作测试的时候不用
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

# 创建神经网络
W1 = tf.Variable(tf.truncated_normal([784,2000],stddev=0.1))
b1 = tf.Variable(tf.zeros([2000])+0.1)
L1 = tf.nn.tanh(tf.matmul(x,W1)+b1)
L1_drop = tf.nn.dropout(L1,keep_prob)

W2 = tf.Variable(tf.truncated_normal([2000,2000],stddev=0.1))
b2 = tf.Variable(tf.zeros([2000])+0.1)
L2 = tf.nn.tanh(tf.matmul(L1_drop,W2)+b2)
L2_drop = tf.nn.dropout(L2,keep_prob)

W3 = tf.Variable(tf.truncated_normal([2000,1000],stddev=0.1))
b3 = tf.Variable(tf.zeros([1000])+0.1)
L3 = tf.nn.tanh(tf.matmul(L2_drop,W3)+b3)
L3_drop = tf.nn.dropout(L3,keep_prob)

W4 = tf.Variable(tf.truncated_normal([1000,10],stddev=0.1))
b4 = tf.Variable(tf.zeros([10])+0.1)
prediction = tf.nn.softmax(tf.matmul(L3_drop,W4)+b4)
# 定义二次代价函数
# loss = tf.reduce_mean(tf.square(y-prediction))
loss =tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=prediction))
# 梯度下降法
train_step = tf.train.GradientDescentOptimizer(0.2).minimize(loss)
# 初始化变量
init = tf.global_variables_initializer()
# 求准确率    求最大的值是在那个位置
score = tf.equal(tf.argmax(y,1),tf.argmax(prediction,1))
# 将true转换成1将false转换成0
accuracy = tf.reduce_mean(tf.cast(score,tf.float32))
with tf.Session() as sess:
    sess.run(init)
    for epoch in range(31):
        for batch in range(m_batch):
            batch_xs,batch_ys = mnist.train.next_batch(batch_size)
            sess.run(train_step,feed_dict={x:batch_xs,y:batch_ys,keep_prob:0.7})

        test_acc = sess.run(accuracy,feed_dict={x:mnist.test.images,y:mnist.test.labels,keep_prob:1.0 })
        train_acc = sess.run(accuracy, feed_dict={x: mnist.train.images, y: mnist.train.labels, keep_prob: 1.0})
        print("当前周期是："+str(epoch)+"  测试准确率："+str(test_acc)+"  测试准确率："+str(train_acc))


"""
当前周期是：0  测试准确率：0.949  测试准确率：0.95845455
当前周期是：1  测试准确率：0.9608  测试准确率：0.97418183
当前周期是：2  测试准确率：0.9648  测试准确率：0.9818909
当前周期是：3  测试准确率：0.9661  测试准确率：0.9861636
当前周期是：4  测试准确率：0.9684  测试准确率：0.9883636
当前周期是：5  测试准确率：0.9685  测试准确率：0.98981816
当前周期是：6  测试准确率：0.9693  测试准确率：0.9906727
当前周期是：7  测试准确率：0.9687  测试准确率：0.9914182
当前周期是：8  测试准确率：0.9696  测试准确率：0.9918909
当前周期是：9  测试准确率：0.9696  测试准确率：0.9923273
当前周期是：10  测试准确率：0.9696  测试准确率：0.9927273
当前周期是：11  测试准确率：0.9699  测试准确率：0.9930909
当前周期是：12  测试准确率：0.9712  测试准确率：0.9935091
当前周期是：13  测试准确率：0.971  测试准确率：0.9937818
当前周期是：14  测试准确率：0.9716  测试准确率：0.9940364
当前周期是：15  测试准确率：0.9708  测试准确率：0.99432725
当前周期是：16  测试准确率：0.9706  测试准确率：0.9945091
当前周期是：17  测试准确率：0.9714  测试准确率：0.99465454
当前周期是：18  测试准确率：0.9709  测试准确率：0.9947091
当前周期是：19  测试准确率：0.9717  测试准确率：0.99472725
当前周期是：20  测试准确率：0.9725  测试准确率：0.99487275
当前周期是：21  测试准确率：0.9727  测试准确率：0.9949818
当前周期是：22  测试准确率：0.972  测试准确率：0.9951091
当前周期是：23  测试准确率：0.9717  测试准确率：0.9951091
当前周期是：24  测试准确率：0.9726  测试准确率：0.9951818
当前周期是：25  测试准确率：0.9722  测试准确率：0.9952545
当前周期是：26  测试准确率：0.9725  测试准确率：0.9953273
当前周期是：27  测试准确率：0.9723  测试准确率：0.99536365
当前周期是：28  测试准确率：0.9724  测试准确率：0.99538183
当前周期是：29  测试准确率：0.9723  测试准确率：0.9954
当前周期是：30  测试准确率：0.9729  测试准确率：0.9954182
"""



