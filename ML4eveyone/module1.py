import numpy as np
import tensorflow as tf

xy=np.loadtxt("data-04-zoo.csv", delimiter=",",dtype=np.float32)
x_data = xy[:,0:-1]
y_data = xy[:,[-1]]

nb_classes = 7

X= tf.placeholder(tf.float32, [None, 16])
Y= tf.placeholder(tf.int32, [None, 1])

Y_1Hot = tf.one_hot(Y, nb_classes)
Y_1Hot = tf.reshape(Y_1Hot,[-1, nb_classes])

W= tf.Variable(tf.random_normal([16,nb_classes]), name='weight')
b= tf.Variable(tf.random_normal([nb_classes]), name='bias')
print(W.get_shape, X._shape)

logits = tf.matmul(X, W) + b
hypo = tf.nn.softmax(logits)

cost_i = tf.nn.softmax_cross_entropy_with_logits(logits = logits,labels = Y_1Hot)

cost = tf.reduce_mean(cost_i)
opti = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost_i)

prediction = tf.argmax(hypo, 1)
correct_p = tf.equal(prediction, tf.argmax(Y_1Hot,1))
acuuracy = tf.reduce_mean(tf.cast(correct_p, tf.float32))

with tf.Session() as sess :
    sess.run(tf.global_variables_initializer()) 

    for step in range(2000) : 
        sess.run(opti, feed_dict = {X: x_data, Y: y_data})
        if step % 10 ==0 :
            loss, acc = sess.run([cost, acuuracy], feed_dict = {X: x_data, Y: y_data })
            print ("Step : {:5} \t Loss : {:.3f}\t Acc : {:.2%}".format(step, loss, acc))

    prd = sess.run(prediction, feed_dict = {X: x_data})

    for p,y in zip(prd, y_data.flatten()) :
        print("[{}] Prediction: {} True Y: {}".format(p == int(y), p, int(y)))


