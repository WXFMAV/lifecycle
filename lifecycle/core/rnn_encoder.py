import tensorflow as tf
import numpy as np

batch_size = 10
max_time = 100
dim = 8
num_units = 32
num_attention = 128
num_out = 64

def lstm_encode(X, batch_size, max_time, num_units, num_attention, num_out):
    source_sequence_length = np.ones((batch_size,)) * max_time

    # Build RNN cell
    # encoder_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units, state_is_tuple=False)
    encoder_cell = tf.contrib.rnn.GRUCell(num_units)

    # Run Dynamic RNN
    #   encoder_outpus: [max_time, batch_size, num_units]
    #   encoder_state: [batch_size, num_units]
    encoder_outputs, encoder_state = tf.nn.dynamic_rnn(
        encoder_cell, X,
        dtype=tf.float32, sequence_length=source_sequence_length, time_major=True)

    # attention_states: [batch_size, num_units, max_time]
    attention_states = tf.transpose(encoder_outputs, [1, 2, 0])
    attention_states2 = tf.reshape(attention_states, [batch_size, -1])
    attention_weight = tf.get_variable("attention_weight", [num_units * max_time, num_attention])
    # Create an attention mechanism, [batch_size, num_units]
    attention_outputs = tf.matmul(attention_states2, attention_weight)

    input_for_fc = tf.concat([attention_outputs, encoder_state], axis=1)
    final_output = tf.contrib.layers.fully_connected(
        inputs=input_for_fc,
        num_outputs=num_out,
        scope='layer',
        reuse=False,
        activation_fn=tf.nn.relu)

    return final_output


x = np.random.rand(max_time, batch_size, dim)
w = np.random.rand(dim, 1)
y = np.dot(np.sum(x, 0), w)

X = tf.placeholder(tf.float32, shape=(max_time, batch_size, dim), name="X")
Y = tf.placeholder(tf.float32, shape=(batch_size, 1), name="Y")

out_lstm_encode = lstm_encode(X, batch_size, max_time, num_units, num_attention, num_out)
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