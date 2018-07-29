# _*_coding:utf-8 _*_
"""
@Time    :2018/6/25 15:17
@Author  :weicm
#@Software: PyCharm
# tensorboard可视化
"""

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib.tensorboard.plugins import projector

# 载入数据集
mnist = input_data.read_data_sets("MNIST_data/",one_hot=True)
# 运行次数
max_steps = 1001
# 图片数量
image_num = 3000
# 文件路径
DIR = "C:/Users/weicm/PycharmProjects/TensorFlow/"
# 定义会话
sess = tf.Session()
# 载入图片 stack打包
embedding = tf.Variable(tf.stack(mnist.test.images[:image_num]),trainable=False,name="embedding")
# 参数概要 计算参数值


def variable_summaries(var):
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        # 记录值并给予名字s
        tf.summary.scalar('mean',mean)  # 平均值
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var-mean)))
        tf.summary.scalar('stddev',stddev) # 标准差
        tf.summary.scalar('max', tf.reduce_max(var)) # 最大值
        tf.summary.scalar('min', tf.reduce_min(var)) # 最小值
        tf.summary.histogram('histogram', var) # 直方图

# 可视化需要定义命名空间
with tf.name_scope('input'):
    # 定义两个占位符
    x = tf.placeholder(tf.float32,[None,784],name="x-input")
    y = tf.placeholder(tf.float32,[None,10],name="y-input")

# 显示图片
with tf.name_scope('input_reshape'):
    # -1代表不确定的值  28行28列  维度是1彩色的维度是3
    image_shaped_input = tf.reshape(x,[-1,28,28,1])
    # 每次放市长图片
    tf.summary.image('input',image_shaped_input,10)

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
    # 定义交叉熵代价函数
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=prediction))
    tf.summary.scalar('loss',loss)
with tf.name_scope('train'):
    # 梯度下降法
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(loss)
# 初始化变量
sess.run(tf.global_variables_initializer())

with tf.name_scope('accuracy'):
    # 求准确率    求最大的值是在那个位置
    with tf.name_scope('score'):
        score = tf.equal(tf.argmax(y,1),tf.argmax(prediction,1))
    # 将true转换成1将false转换成0
    with tf.name_scope('accuracy'):
        accuracy = tf.reduce_mean(tf.cast(score,tf.float32))
        tf.summary.scalar('accuracy',accuracy)

# 产生metadata文件
if tf.gfile.Exists(DIR+'projector/projector/metadata.tsv'):
    # 检查是否存在该文件  有的话就直接删除
    tf.gfile.DeleteRecursively(DIR+'projector/projector/metadata.tsv')
with open(DIR+'projector/projector/metadata.tsv','w') as f:
    # 拿到测试集标签   标签在那个位置哪里为1  其余为0
    labels = sess.run(tf.argmax(mnist.test.labels[:],1))
    for i in range(image_num):
        f.write(str(labels[i]+'\n'))

# 合并所有的summary
merged = tf.summary.merge_all()
# 定义writer
projector_writer = tf.summary.FileWriter(DIR+'projector/projector',sess.graph)
# 用来保存网络模型
saver = tf.train.Saver()
# 定义配置项
config = projector.ProjectorConfig()
# 设置
embed = config.embeddings.add()
# 图片名字赋值
embed.tensor_name = embedding.name
embed.metadata_path = DIR+'projector/projector/metadata.tsv'
embed.sprite.image_path = DIR+'projector/data/mnist_10k_sprite.png'
# 图片按照28*28进行切分
embed.sprite.single_image_dim.extend([28,28])
# 可视化工具
projector.visualize_embeddings(projector_writer,config)

for i in range(max_steps):
    # 每个批次100个样本
    batch_xs, batch_ys = mnist.train.next_batch(100)
    # 固定配置
    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    run_metadata = tf.RunMetadata()
    summary,_ = sess.run([merged,train_step],feed_dict={x:batch_xs,y:batch_ys},options=run_options,run_metadata=run_metadata)
    projector_writer.add_run_metadata(run_metadata,'step%03d'%i)
    projector_writer.add_summary(summary,i)

    if i%100 == 0:
        acc = sess.run(accuracy,feed_dict={x:mnist.test.images,y:mnist.test.labels})
        print("Iter"+str(i)+",Testing Accuracy="+str(acc))

saver.save(sess,DIR+'projector/projrctor/a_model.ckpt',global_step=max_steps)
projector_writer.close()
sess.close()
