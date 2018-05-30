from copy import copy
from functools import reduce

import tensorflow.contrib as tc

from util import *


class DQN(object):
    def __init__(self, name, value_net, memory, observation_shape, action_shape, action_num,
                 param_noise=None, action_noise=None,
                 gamma=0.99, tau=0.001, normalize_returns=False, enable_popart=False, normalize_observations=False,
                 batch_size=128, observation_range=(-5., 5.), action_range=(-1., 1.), return_range=(-np.inf, np.inf),
                 critic_l2_reg=0., learning_rate=1e-3, clip_norm=None, reward_scale=1.,
                 ckp_dir=None, global_step=None,
                 obs0=None, obs1=None, terminals1=None, rewards=None, actions=None, action_sets=None):
        self.name = name
        # Inputs.
        self.obs0 = tf.placeholder(tf.float32, shape=(None,) + observation_shape, name='obs0') if obs0 is None else obs0
        self.obs1 = tf.placeholder(tf.float32, shape=(None,) + observation_shape, name='obs1') if obs1 is None else obs1
        self.terminals1 = tf.placeholder(tf.float32, shape=(None, 1), name='terminals1') if terminals1 is None else terminals1
        self.rewards = tf.placeholder(tf.float32, shape=(None, 1), name='rewards') if rewards is None else rewards
        self.actions = tf.placeholder(tf.float32, shape=(None,) + action_shape, name='actions') if actions is None else actions
        # the placeholder for action set
        # self.action_sets = tf.placeholder(tf.float32, shape=(None,) + (action_num,) + (action_num,), name='action_sets') \
        #     if action_sets is None else action_sets
        self.action_sets = [tf.placeholder(tf.float32, shape=(None,) + (action_num,), name=str(i)+'_action_code')
                            for i in range(action_num)] if action_sets is None else action_sets

        # Parameters.
        self.gamma = gamma
        self.tau = tau
        self.memory = memory
        self.normalize_observations = normalize_observations
        self.normalize_returns = normalize_returns
        self.action_noise = action_noise
        self.param_noise = param_noise
        self.action_range = action_range
        self.return_range = return_range
        self.observation_range = observation_range
        self.value_net = value_net
        self.learning_rate = learning_rate
        self.clip_norm = clip_norm
        self.enable_popart = enable_popart
        self.reward_scale = reward_scale
        self.batch_size = batch_size
        self.stats_sample = None
        self.critic_l2_reg = critic_l2_reg
        self.ckp_dir = ckp_dir
        self.global_step = global_step
        self.action_num = action_num

        # Observation normalization.
        if self.normalize_observations:
            with tf.variable_scope('obs_rms'):
                self.obs_rms = None  # RunningMeanStd(shape=observation_shape)
        else:
            self.obs_rms = None
        normalized_obs0 = tf.clip_by_value(normalize(self.obs0, self.obs_rms),
                                           self.observation_range[0], self.observation_range[1])
        normalized_obs1 = tf.clip_by_value(normalize(self.obs1, self.obs_rms),
                                           self.observation_range[0], self.observation_range[1])

        # Return normalization.
        if self.normalize_returns:
            with tf.variable_scope('ret_rms'):
                self.ret_rms = None  # RunningMeanStd()
        else:
            self.ret_rms = None

        # Create target value network
        target_value_net = copy(value_net)
        target_value_net.name = 'target' + value_net.name
        self.target_value_net = target_value_net

        # Define the Q(s,a) and maxQ(s',a') operation
        self.Q_obs0_op = self.value_net(self.obs0, self.actions)
        self.targeQ_obs0_op = self.target_value_net(self.obs0, self.actions)
        self.maxQ_obs1_op = tf.reduce_max([self.target_value_net(self.obs1, self.action_sets[i], reuse=True)
                                           for i in range(action_num)])
        self.target_Q_op = self.rewards + (1.0 - self.terminals1) * gamma * self.maxQ_obs1_op

        # default value for action set placeholder
        self.action_set_value = []
        for act in range(0, action_num):
            act_code = self.generate_action_code(act, action_num)
            self.action_set_value.append(act_code)

        # set up network optimizer and target network updates
        self.setup_value_network_optimizer()
        self.setup_target_network_updates()

        # add metrics
        self.metrics = []
        with tf.name_scope(name + '_training'):
            self.metrics.append(tf.summary.scalar('MSBE', self.q_loss))
            self.metrics.append(tf.summary.scalar('reference_Q_mean', tf.reduce_mean(self.Q_obs0_op)))

        with tf.name_scope(name + '_input'):
            # self.metrics.append(tf.summary.scalar('action_avg', tf.reduce_mean(self.actions)))
            # self.metrics.append(tf.summary.scalar('reward_avg', tf.reduce_mean(self.rewards)))
            # self.metrics.append(tf.summary.histogram('obs', self.obs0))
            # self.metrics.append(tf.summary.histogram('action', self.actions))
            self.metrics.append(tf.summary.scalar('reward_avg', tf.reduce_mean(self.rewards)))
            self.metrics.append(tf.summary.histogram('obs', self.obs0))
        self.merged = tf.summary.merge(self.metrics)

        # saver and init
        self.saver = tf.train.Saver()
        self.init = tf.global_variables_initializer()

        # for feed
        self.obs1_val = np.random.rand(1, self.obs1.get_shape()[1].value)
        self.rewards_val = np.random.rand(1, self.rewards.get_shape()[1].value)
        self.terminals1_val = np.random.rand(1, self.terminals1.get_shape()[1].value)
        self.obs0_val = np.random.rand(1, self.obs0.get_shape()[1].value)
        self.actions_val = np.random.rand(1, self.actions.get_shape()[1].value)

    def setup_target_network_updates(self):
        qnet_init_updates, qnet_soft_updates = get_target_updates(self.value_net.trainable_vars,
                                                                  self.target_value_net.trainable_vars, self.tau)
        self.target_init_updates = qnet_init_updates
        self.target_soft_updates = qnet_soft_updates

    def setup_value_network_optimizer(self):
        print('setting up value network optimizer')
        self.q_loss = tf.reduce_mean(tf.square(self.Q_obs0_op - self.target_Q_op))
        if self.critic_l2_reg > 0.:
            critic_reg_vars = [var for var in self.value_net.trainable_vars if
                               'kernel' in var.name and 'output' not in var.name]
            for var in critic_reg_vars:
                print('  regularizing: {}'.format(var.name))
            print('  applying l2 regularization with {}'.format(self.critic_l2_reg))
            critic_reg = tc.layers.apply_regularization(
                tc.layers.l2_regularizer(self.critic_l2_reg),
                weights_list=critic_reg_vars
            )
            self.q_loss += critic_reg
        critic_shapes = [var.get_shape().as_list() for var in self.value_net.trainable_vars]
        critic_nb_params = sum([reduce(lambda x, y: x * y, shape) for shape in critic_shapes])
        print('  critic shapes: {}'.format(critic_shapes))
        print('  critic params: {}'.format(critic_nb_params))
        self.q_grads = tf.gradients(self.q_loss, self.value_net.trainable_vars)
        if self.clip_norm is not None:
            # self.critic_grads, self.critic_global_norm = tf.clip_by_global_norm(self.critic_grads, self.clip_norm)
            self.q_grads = [tf.clip_by_norm(x, self.clip_norm) for x in self.q_grads]
        self.q_optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.9, beta2=0.999,
                                                  epsilon=1e-08)
        self.q_train_op = self.q_optimizer.apply_gradients(zip(self.q_grads, self.value_net.trainable_vars),
                                                           global_step=self.global_step)

    def pi(self, obs, apply_noise=True, compute_Q=True):
        actor_tf = self.actor_tf
        feed_dict = {self.obs0: [obs]}
        if compute_Q:
            action, q = self.sess.run([actor_tf, self.critic_with_actor_tf], feed_dict=feed_dict)
        else:
            action = self.sess.run(actor_tf, feed_dict=feed_dict)
            q = None
        action = action.flatten()
        if self.action_noise is not None and apply_noise:
            noise = self.action_noise()
            assert noise.shape == action.shape
            action += noise
        action = np.clip(action, self.action_range[0], self.action_range[1])
        return action, q

    def store_transition(self, obs0, action, reward, obs1, terminal1):
        reward *= self.reward_scale

        # we sample pv samples
        # also consider ipv samples
        store = True
        if not terminal1 and reward < 1e-5:
            import random
            if 'android' in self.name:
                if random.random() > 0.3:
                    store = False
            else:
                if random.random() > 0.3:
                    store = False
        else:
            store = True

        if store:
            # transfer the action to action code
            action_code = self.generate_action_code(action, self.action_num)
            self.memory.append(obs0, action_code, reward, obs1, terminal1)
            if self.normalize_observations:
                self.obs_rms.update(np.array([obs0]))

    def train(self):
        # Get a batch.
        batch = self.memory.sample(batch_size=self.batch_size)

        # the dict for action set place holders
        feed_dict = {}
        for i in range(self.action_num):
            feed_dict[self.action_sets[i]] = [self.action_set_value[i] for j in range(self.batch_size)]
        feed_dict[self.obs1] = batch['obs1']
        feed_dict[self.rewards] = batch['rewards']
        feed_dict[self.terminals1] = batch['terminals1'].astype('float32')
        feed_dict[self.obs0] = batch['obs0']
        feed_dict[self.actions] = batch['actions']

        ops = [self.merged, self.target_soft_updates, self.Q_obs0_op, self.target_Q_op, self.q_train_op]
        self.summary, _, Qsa, Qspap, _ = self.sess.run(ops, feed_dict=feed_dict)

        # self.summary, _, _, _, _ = self.sess.run(ops, feed_dict={
        #     self.obs1: batch['obs1'],
        #     self.rewards: batch['rewards'],
        #     self.terminals1: batch['terminals1'].astype('float32'),
        #     self.obs0: batch['obs0'],
        #     self.actions: batch['actions']
        # })

    def write_summary(self, step):
        self.writer.add_summary(self.summary, step)

    def save_model(self):
        save_path = self.saver.save(self.sess, self.ckp_dir + '/model')
        print("Model saved in file: %s" % save_path)

    def initialize(self, sess, init_var=False):
        self.sess = sess
        if init_var:
            self.sess.run(self.init)
            self.writer = tf.summary.FileWriter(self.ckp_dir, self.sess.graph)

        # the dict for action set place holders
        feed_dict = {}
        for i in range(self.action_num):
            feed_dict[self.action_sets[i]] = [self.action_set_value[i]]
        feed_dict[self.obs1] = self.obs1_val
        feed_dict[self.rewards] = self.rewards_val
        feed_dict[self.terminals1] = self.terminals1_val
        feed_dict[self.obs0] = self.obs0_val
        feed_dict[self.actions] = self.actions_val
        self.sess.run(self.target_init_updates, feed_dict=feed_dict)

        # self.sess.run(self.target_init_updates, feed_dict={
        #     self.obs1: self.obs1_val,
        #     self.rewards: self.rewards_val,
        #     self.terminals1: self.terminals1_val,
        #     self.obs0: self.obs0_val,
        #     self.actions: self.actions_val
        # })

    def update_target_net(self):

        # the dict for action set place holders
        feed_dict = {}
        for i in range(self.action_num):
            feed_dict[self.action_sets[i]] = [self.action_set_value[i]]

        feed_dict[self.obs1] = self.obs1_val
        feed_dict[self.rewards] = self.rewards_val
        feed_dict[self.terminals1] = self.terminals1_val
        feed_dict[self.obs0] = self.obs0_val
        feed_dict[self.actions] = self.actions_val
        self.sess.run(self.target_soft_updates, feed_dict=feed_dict)

        # self.sess.run(self.target_soft_updates, feed_dict={
        #     self.obs1: self.obs1_val,
        #     self.rewards: self.rewards_val,
        #     self.terminals1: self.terminals1_val,
        #     self.obs0: self.obs0_val,
        #     self.actions: self.actions_val
        # })

    def reset(self):
        # Reset internal state after an episode is complete.
        if self.action_noise is not None:
            self.action_noise.reset()

    def generate_action_code(self, act_index, action_num):
        action_code = []
        for index in range(0, action_num):
            if index == act_index:
                action_code.append(1)
            else:
                action_code.append(0)

        return action_code

