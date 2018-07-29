# _*_coding:utf-8 _*_
"""
@Time    :2018/6/24 13:51
@Author  :weicm
#@Software: PyCharm
"""

import tensorflow as tf
# 创建常量op
ml = tf.constant([[3,3]])
m2 = tf.constant([[2],[3]])
# 创建矩阵乘法op
product = tf.matmul(ml,m2)
# print(product)      # Tensor("MatMul:0", shape=(1, 1), dtype=int32) 需要放到图中执行
# 定义会话启动默认的图
sess = tf.Session()
# 使用run方法来执行乘法op
result = sess.run(product)
print(result)    # [[15]]
sess.close()

# 不需要关闭方法
with tf.Session() as sess:
    sess = tf.Session()
    # 使用run方法来执行乘法op
    result = sess.run(product)
    print(result)
