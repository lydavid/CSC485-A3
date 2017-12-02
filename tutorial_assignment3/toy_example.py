import tensorflow as tf

y = tf.placeholder(tf.float32, (None, ))
x = tf.placeholder(tf.float32, (None, ))

m = tf.Variable(0.0, dtype=tf.float32)
c = tf.constant(1.0, dtype=tf.float32)

y_ = m*x + c

loss = tf.reduce_mean(tf.square(y_ - y))
optimizer = tf.train.GradientDescentOptimizer(0.1)
train = optimizer.minimize(loss)

with tf.Session() as sess:
    # Initialize our variables
    init = tf.global_variables_initializer()
    sess.run(init)
    
    wrong_answer = sess.run(y_, {x: [4]})
    
    for i in range(10):
        sess.run(train, {y: [3,5,7], x: [1,2,3]})

    right_answer = sess.run(y_, {x: [4]})

    print(wrong_answer, right_answer)
