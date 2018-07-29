# _*_coding:utf-8 _*_
"""
@Time    :2018/6/24 20:03
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
loss = tf.reduce_mean(tf.square(y-prediction))
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
    for epoch in range(21):
        for batch in range(m_batch):
            batch_xs,batch_ys = mnist.train.next_batch(batch_size)
            sess.run(train_step,feed_dict={x:batch_xs,y:batch_ys})

        acc = sess.run(accuracy,feed_dict={x:mnist.test.images,y:mnist.test.labels})
        print("当前周期是："+str(epoch)+"  准确率："+str(acc))


"""
当前周期是：0  准确率：0.8331
当前周期是：1  准确率：0.8707
当前周期是：2  准确率：0.8803
当前周期是：3  准确率：0.8886
当前周期是：4  准确率：0.8941
当前周期是：5  准确率：0.8975
当前周期是：6  准确率：0.9008
当前周期是：7  准确率：0.902
当前周期是：8  准确率：0.9038
当前周期是：9  准确率：0.9049
当前周期是：10  准确率：0.9063
当前周期是：11  准确率：0.9068
当前周期是：12  准确率：0.9079
当前周期是：13  准确率：0.9088
当前周期是：14  准确率：0.9093
当前周期是：15  准确率：0.9107
当前周期是：16  准确率：0.9119
当前周期是：17  准确率：0.9117
当前周期是：18  准确率：0.9135
当前周期是：19  准确率：0.9132
当前周期是：20  准确率：0.9133
"""