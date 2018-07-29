# _*_coding:utf-8 _*_
"""
@Time    :2018/6/24 14:43
@Author  :weicm
#@Software: PyCharm
"""

import tensorflow as tf

# x = tf.Variable([1,2])
# a = tf.constant([3,3])
# # 增加一个减法op
# sub = tf.subtract(x,a)
# # 增加一个加法op
# add = tf.add(x,sub)
# init = tf.global_variables_initializer()
# with tf.Session() as sess:
#     sess.run(init)
#     result = sess.run(add)
#     print(result)

# 创建一个变量使得其可以循环加1
# 创建一个变量初始化为0
state = tf.Variable(0,name="counter")
# 创建一个op作用是使state加1
new_state = tf.add(state,1)
# 赋值操作
update = tf.assign(state,new_state)
# 赋值初始化
init = tf.global_variables_initializer()
with tf.Session()  as sess:
    sess.run(init)
    print(sess.run(state))
    for _ in range(5):
        sess.run(update)
        print(sess.run(state))