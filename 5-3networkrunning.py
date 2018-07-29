# _*_coding:utf-8 _*_
"""
@Time    :2018/6/25 14:29
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

# 参数概要 计算参数值
def variable_summaries(var):
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        # 记录值并给予名字s
        tf.summary.scalar('mean',mean)  #平均值
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var-mean)))
        tf.summary.scalar('stddev',stddev) #标准差
        tf.summary.scalar('max', tf.reduce_max(var)) # 最大值
        tf.summary.scalar('min', tf.reduce_min(var)) # 最小值
        tf.summary.histogram('histogram', var) # 直方图


# 可视化需要定义命名空间
with tf.name_scope('input'):
    # 定义两个占位符
    x = tf.placeholder(tf.float32,[None,784],name="x-input")
    y = tf.placeholder(tf.float32,[None,10],name="y-input")

with tf.name_scope('layer'):
    # 创建神经网络
    with tf.name_scope('weight'):
        W = tf.Variable(tf.zeros([784,10]),name="W")
        # 在网络实际运行中需要观察权值变化
        variable_summaries(W)
    with tf.name_scope('biases'):
        b = tf.Variable(tf.zeros([10]),name='b')
        variable_summaries(b)
    with tf.name_scope('wx_plus_b'):
        wx_plus_b = tf.matmul(x,W)+b
    with tf.name_scope('softmax'):
        prediction = tf.nn.softmax(wx_plus_b)
with tf.name_scope('loss'):
    # 定义二次代价函数
    loss = tf.reduce_mean(tf.square(y-prediction))
    tf.summary.scalar('loss',loss)
with tf.name_scope('train'):
    # 梯度下降法
    train_step = tf.train.GradientDescentOptimizer(0.2).minimize(loss)
# 初始化变量
init = tf.global_variables_initializer()
with tf.name_scope('accuracy'):
    # 求准确率    求最大的值是在那个位置
    with tf.name_scope('score'):
        score = tf.equal(tf.argmax(y,1),tf.argmax(prediction,1))
    # 将true转换成1将false转换成0
    with tf.name_scope('accuracy'):
        accuracy = tf.reduce_mean(tf.cast(score,tf.float32))
        tf.summary.scalar('accuracy',accuracy)
# 合并所有的suammary
merged = tf.summary.merge_all()
with tf.Session() as sess:
    sess.run(init)
    # 存放在当前路径下  如果没有次路径则新建
    # 存好以后打开命令提示符对应到相应盘符下通过
    # 命令 tensorboard --logdir=C:\Users\weicm\PycharmProjects\TensorFlow\logs 得到相应网址
    writer = tf.summary.FileWriter('logs/',sess.graph)
    for epoch in range(51):
        for batch in range(m_batch):
            batch_xs,batch_ys = mnist.train.next_batch(batch_size)
            # 将merged返回值传到summary中
            summary,_ = sess.run([merged,train_step],feed_dict={x:batch_xs,y:batch_ys})
        # 将summary和运行的周期写到文件中
        writer.add_summary(summary,epoch)
        acc = sess.run(accuracy,feed_dict={x:mnist.test.images,y:mnist.test.labels})
        print("当前周期是："+str(epoch)+"  准确率："+str(acc))