import tensorflow as tf
import tensorflow.contrib as tc
import numpy as np

def lstm_encode(X):
#    print('X get shape:', X.get_shape().as_list())
#    batch_size = None #X.get_shape()[0]
    max_time = int(X.get_shape().as_list()[1])
    num_units = int(X.get_shape().as_list()[2])
    
    num_attention = 128
    num_out = 64
    #source_sequence_length = np.ones((batch_size,)) * max_time

    # Build RNN cell
    # encoder_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units, state_is_tuple=False)
    encoder_cell = tf.contrib.rnn.GRUCell(num_units)

    # Run Dynamic RNN
    #   encoder_outpus: [max_time, batch_size, num_units]
    #   encoder_state: [batch_size, num_units]
    encoder_outputs, encoder_state = tf.nn.dynamic_rnn(
         encoder_cell, X,
    #     dtype=tf.float32, sequence_length=source_sequence_length, time_major=True)
         dtype=tf.float32, time_major=False)

    # attention_states: [batch_size, max_time, num_units]
    attention_states = tf.transpose(encoder_outputs, [0, 1, 2])
    attention_states2 = tf.reshape(attention_states, [-1, attention_states.get_shape()[1].value * attention_states.get_shape()[2].value])
    attention_weight = tf.get_variable("attention_weight", [attention_states.get_shape()[1].value * attention_states.get_shape()[2].value, num_attention])
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


def cnn_encode(X, reuse=False):

    max_time = int(X.get_shape().as_list()[1])
    num_units = int(X.get_shape().as_list()[2])
    num_out = 64

    with tf.variable_scope('cnn', reuse=reuse) as scope:
        def weight_varible(shape, name):
            if not tf.get_variable_scope().reuse :
                initial = tf.truncated_normal_initializer(mean=0.0, stddev=0.1)
                return tf.get_variable(shape=shape, initializer=initial, name=name)

            else:
                return tf.get_variable(initializer=tf.get_variable_scope().reuse_variables(), name=name)

        def bias_variable(shape, name):
            if not tf.get_variable_scope().reuse:
                initial = tf.constant_initializer(0.1)
                return tf.get_variable(shape=shape, initializer=initial, name=name)
            else:
                return tf.get_variable(initializer=tf.get_variable_scope().reuse_variables(), name=name)
    
        def conv2d(x, W):
            return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
    
        def max_pool_2x2(x):
            return tf.nn.max_pool(x, ksize=[1, 40, 1, 1], strides=[1, 40, 1, 1], padding='SAME')
    
        X_t = tf.transpose(X, [0, 1, 2]) # [batchsize, max_time, num_units]
        x_image = tf.reshape(X_t, [-1, max_time, num_units, 1])
    
        # paras
        W_conv1 = weight_varible([30, 5, 1, 64], 'w1')
        b_conv1 = bias_variable([64],'b1')
    
        h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
        h_pool1 = max_pool_2x2(h_conv1)
    
        # conv layer-2
        W_conv2 = weight_varible([30, 5, 64, 32], 'w2')
        b_conv2 = bias_variable([32], 'b2')
    
        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
        h_pool2 = max_pool_2x2(h_conv2)
    
        # full connection
        W_fc1 = weight_varible([h_pool2.get_shape()[1].value * h_pool2.get_shape()[2].value * h_pool2.get_shape()[3].value, num_out], 'w3')
        b_fc1 = bias_variable([num_out], 'b3')
    
        h_pool2_flat = tf.reshape(h_pool2, [-1, h_pool2.get_shape()[1].value * h_pool2.get_shape()[2].value * h_pool2.get_shape()[3].value])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1) # [batchsize, num_out]
#        print('Name::::', W_conv1.name, b_conv1.name, W_conv2.name, b_conv2.name, W_fc1.name, b_fc1.name)
#        print(W_conv1.name.find('critic_1'))
#        assert(W_conv1.name.find('critic_1') == -1)
    return h_fc1


class Act(object):
    def __init__(self, name, alpha=0.2):
        def lrelu(i):
            negative = tf.nn.relu(-i)
            res = i + negative * (1.0-alpha)
            return res

        activations = {
            'relu': tf.nn.relu,
            'tanh': tf.tanh,
            'sigmoid': tf.sigmoid,
            'softmax': tf.nn.softmax,
            'elu': tf.nn.elu,
            'lrelu': lrelu,
            'softplus': tf.nn.softplus,
        }

        self.func = activations[name]

    def __call__(self, i, *args, **kwargs):
        return self.func(i, *args, **kwargs)


class Model(object):
    def __init__(self, name):
        self.name = name

    @property
    def vars(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)

    @property
    def trainable_vars(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

    @property
    def perturbable_vars(self):
        return [var for var in self.trainable_vars if 'LayerNorm' not in var.name]


class QNet(Model):
    def __init__(self, net_config, name='qnet', layer_norm=True):
        super(QNet, self).__init__(name=name)
        self.net_config = net_config
        self.layer_norm = layer_norm

    def __call__(self, obs, act_code, reuse=False):
        # with tf.variable_scope(self.name) as scope:
        #     if reuse:
        #         scope.reuse_variables()

            # for i in range(len(self.net_config) - 1):
            #     x = tf.layers.dense(x, self.net_config[i][0])
            #     if self.layer_norm:
            #         x = tc.layers.layer_norm(x, center=True, scale=True)
            #     act = Act(self.net_config[i][1])
            #     x = act(x)
            #
            # x = tf.layers.dense(x, self.net_config[-1][0],
            #                     kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3))
            # act = Act(self.net_config[-1][1])
            # x = act(x)

        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()

            x = tf.concat([obs, act_code], 1)

            for i in range(0, len(self.net_config)):
                x = tf.layers.dense(x, self.net_config[i][0])
                if self.layer_norm:
                    x = tc.layers.layer_norm(x, center=True, scale=True)
                act = Act(self.net_config[i][1])
                x = act(x)

            x = tf.layers.dense(x, 1, kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3))
        return x


class Reg(Model):
    def __init__(self, net_config, name='reg', layer_norm=True):
        super(Reg, self).__init__(name=name)
        self.net_config = net_config
        self.layer_norm = layer_norm

    def __call__(self, obs, action, reuse=False):
        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()

            # x = obs
            # concat the state and action in the input layer
            x = tf.concat([obs, action], axis=-1)

            x = tf.layers.dense(x, self.net_config[0][0])
            if self.layer_norm:
                x = tc.layers.layer_norm(x, center=True, scale=True)
            act = Act(self.net_config[0][1])
            x = act(x)
            # also concat the first hidden layer and action
            x = tf.concat([x, action], axis=-1)

            for i in range(1, len(self.net_config)):
                x = tf.layers.dense(x, self.net_config[i][0])
                if self.layer_norm:
                    x = tc.layers.layer_norm(x, center=True, scale=True)
                act = Act(self.net_config[i][1])
                x = act(x)

            x = tf.layers.dense(x, 1, kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3))
        return x


class Actor(Model):
    def __init__(self, net_config, name='actor', layer_norm=True):
        super(Actor, self).__init__(name=name)
        self.net_config = net_config
        self.layer_norm = layer_norm

    def __call__(self, obs, reuse=False, cnn_reuse=False):
        print("actor---" + str(cnn_reuse))
        if len(obs.get_shape().as_list()) == 3:
            x = cnn_encode(obs, cnn_reuse)
        else:
            x = obs

        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()

            for i in range(len(self.net_config)-1):
                x = tf.layers.dense(x, self.net_config[i][0])
                if self.layer_norm:
                    x = tc.layers.layer_norm(x, center=True, scale=True)
                act = Act(self.net_config[i][1])
                x = act(x)
            
            x = tf.layers.dense(x, self.net_config[-1][0], kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3))
            act = Act(self.net_config[-1][1])
            x = act(x)
        return x


class Critic(Model):
    def __init__(self, net_config, name='critic', layer_norm=True):
        super(Critic, self).__init__(name=name)
        self.net_config = net_config
        self.layer_norm = layer_norm

    def __call__(self, obs, action, reuse=False, cnn_reuse=False):
        print("critic---" + str(cnn_reuse))
        if len(obs.get_shape().as_list()) == 3:
            x = cnn_encode(obs, cnn_reuse)
        else:
            x = obs

        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()

            x = tf.layers.dense(x, self.net_config[0][0])
            if self.layer_norm:
                x = tc.layers.layer_norm(x, center=True, scale=True)
            act = Act(self.net_config[0][1])
            x = act(x)
            print('--------:', obs.shape, action.shape)
            x = tf.concat([x, action], axis=-1)

            for i in range(1, len(self.net_config)):
                x = tf.layers.dense(x, self.net_config[i][0])
                if self.layer_norm:
                    x = tc.layers.layer_norm(x, center=True, scale=True)
                act = Act(self.net_config[i][1])
                x = act(x)

            x = tf.layers.dense(x, 1, kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3))
        print('----trainable variable __call_')
        for var in self.trainable_vars:
            print(var)
        return x

    @property
    def output_vars(self):
        output_vars = [var for var in self.trainable_vars if 'output' in var.name]
        return output_vars
