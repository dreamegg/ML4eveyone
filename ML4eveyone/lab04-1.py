import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.animation as anim

#import tensorflow.python.debug as tf_debug

'''
tf.set_random_seed(777)

xy=np.loadtxt('data-01-test-score.csv', delimiter=',',dtype=np.float32)

x_data = xy[:, 0:-1]
y_data = xy[:, [-1]]
'''
filename_Q=tf.train.string_input_producer(['data-01-test-score.csv'], shuffle=False, name='filename_Q')
reader = tf.TextLineReader()
key,value = reader.read(filename_Q)

record_default = [[0.], [0.], [0.], [0.]]
xy = tf.decode_csv(value, record_defaults=record_default)

train_x_batch, train_y_batch = tf.train.batch([xy[0:-1], xy[-1:]],batch_size=10)

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

#sess = tf_debug.LocalCLIDebugWrapperSession(sess)
coord=tf.train.Coordinator()
thread = tf.train.start_queue_runners(sess=sess, coord=coord)
print (coord, thread)


out =[]
for step in range (2001) : 
    x_batch, y_batch = sess.run([train_x_batch,train_y_batch])
    if step ==0 :
        print (x_batch, y_batch)
    cost_val, hy_val, _ = sess.run([cost, hypo, train], feed_dict = { x: x_batch, y: y_batch})
    out.append(cost_val)
    if step % 100 == 0 :
        print(step, "Cost:", cost_val) # "Prediction : " ,hy_val)


plt.plot(range (2001),  out)
plt.ylabel('Cost')
plt.show()

coord.request_stop()
coord.join(thread)

print (hypo)
print("1st Score will be" , sess.run(hypo, feed_dict={x:[[100, 70, 101]]}))
print("2nd Score will be" , sess.run(hypo, feed_dict={x:[[60, 70, 110], [90, 100, 80]]}))
