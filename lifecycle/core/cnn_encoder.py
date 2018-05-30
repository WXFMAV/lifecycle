import tensorflow as tf
import numpy as np

batch_size = 10
max_time = 100
dim = 8
num_units = 32
num_attention = 128
num_out = 64

def cnn_encode(X, batch_size, max_time, dim, num_out):
    def weight_varible(shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    def bias_variable(shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    def conv2d(x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    def max_pool_2x2(x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    X_t = tf.transpose(X, [1, 0, 2]) # [batchsize, max_time, dim]
    x_image = tf.reshape(X_t, [-1, max_time, dim, 1])

    # paras
    W_conv1 = weight_varible([5, 5, 1, 32])
    b_conv1 = bias_variable([32])

    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    # conv layer-2
    W_conv2 = weight_varible([5, 5, 32, 64])
    b_conv2 = bias_variable([64])

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    # full connection
    W_fc1 = weight_varible([25 * 2 * 64, num_out])
    b_fc1 = bias_variable([num_out])

    h_pool2_flat = tf.reshape(h_pool2, [-1, 25 * 2 * 64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1) # [batchsize, num_out]

    return h_fc1


x = np.random.rand(max_time, batch_size, dim)

w = np.random.rand(dim, 1)
y = np.dot(np.sum(x, 0), w)

X = tf.placeholder(tf.float32, shape=(max_time, batch_size, dim), name="X")
Y = tf.placeholder(tf.float32, shape=(batch_size, 1), name="Y")

# s = x
out_lstm_encode = cnn_encode(X, batch_size, max_time, dim, num_out)

prediction = tf.contrib.layers.fully_connected(
        inputs=out_lstm_encode,
        num_outputs=1,
        scope='out',
        reuse=False,
        activation_fn=None)

diff = tf.pow(prediction - Y, 2)
square_loss = tf.reduce_mean(diff)

optimizer = tf.train.AdamOptimizer(0.001)
train_op = optimizer.minimize(square_loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())

    for i in range(10000):
        _, loss = sess.run([train_op, square_loss], feed_dict={X: x, Y: y})
        print('iter=%d, loss=%f' % (i, loss))