import numpy as np
import tensorflow as tf

tf.set_random_seed(777)

xy=np.loadtxt('data-01-test-score.csv', delimiter=',',dtype=np.float32)

x_data = xy[:, 0:-1]
y_data = xy[:, [-1]]

x=tf.placeholder(tf.float32, shape=[None,3])
y=tf.placeholder(tf.float32, shape=[None,1])

w= tf.Variable(tf.random_normal([3,1]), name= "weight")
b= tf.Variable(tf.random_normal([1]), name = "bias")

print (w,b)
hypo = tf.matmul(x, w) + b

cost = tf.reduce_mean(tf.square(hypo - y))

opt = tf.train.GradientDescentOptimizer(learning_rate=1e-5)
train = opt.minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range (50000001) : 
    cost_val, hy_val, _ = sess.run([cost, hypo, train], feed_dict = { x: x_data, y: y_data})
    if step % 1000 == 0 :
        print(step, "Cost:", cost_val) # "Prediction : " ,hy_val)
