from copy import copy
from functools import reduce

import tensorflow.contrib as tc
import numpy as np
import tensorflow as tf
from tensorflow import *
from core.util import *

class DDPG(object):
    def __init__(self, name, actor, critic, memory, observation_shape, action_shape, param_noise=None, action_noise=None,
                 gamma=0.99, tau=0.001, normalize_returns=False, enable_popart=False, normalize_observations=False,
                 batch_size=128, observation_range=(-5., 5.), action_range=(-1., 1.), return_range=(-np.inf, np.inf),
                 critic_l2_reg=0., actor_lr=1e-4, critic_lr=1e-3, clip_norm=None, reward_scale=1.,
                 ckp_dir=None, global_step=None,
                 obs0=None, obs1=None, terminals1=None, rewards=None, actions=None):
        self.name = name
        # Inputs.
        self.obs0 = tf.placeholder(tf.float32, shape=(None,) + observation_shape, name='obs0') if obs0 is None else obs0
        self.obs1 = tf.placeholder(tf.float32, shape=(None,) + observation_shape, name='obs1') if obs1 is None else obs1
        self.terminals1 = tf.placeholder(tf.float32, shape=(None, 1), name='terminals1') if terminals1 is None else terminals1
        self.rewards = tf.placeholder(tf.float32, shape=(None, 1), name='rewards') if rewards is None else rewards
        self.actions = tf.placeholder(tf.float32, shape=(None,) + action_shape, name='actions') if actions is None else actions

        # self.obs0 = tf.placeholder(tf.float32, shape=(None,) + observation_shape, name=name+'_obs0') if obs0 is None else obs0
        # self.obs1 = tf.placeholder(tf.float32, shape=(None,) + observation_shape, name=name+'_obs1') if obs1 is None else obs1
        # self.terminals1 = tf.placeholder(tf.float32, shape=(None, 1), name=name+'_terminals1') if terminals1 is None else terminals1
        # self.rewards = tf.placeholder(tf.float32, shape=(None, 1), name=name+'_rewards') if rewards is None else rewards
        # self.actions = tf.placeholder(tf.float32, shape=(None,) + action_shape, name=name+'_actions') if actions is None else actions

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
        self.critic = critic
        self.actor = actor
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.clip_norm = clip_norm
        self.enable_popart = enable_popart
        self.reward_scale = reward_scale
        self.batch_size = batch_size
        self.stats_sample = None
        self.critic_l2_reg = critic_l2_reg
        self.ckp_dir = ckp_dir
        self.global_step = global_step

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

        # Create target networks.
        target_actor = copy(actor)
        target_actor.name = 'target' + actor.name
        self.target_actor = target_actor
        target_critic = copy(critic)
        target_critic.name = 'target' + critic.name
        self.target_critic = target_critic
        print('-----len:', len(critic.trainable_vars), len(target_critic.trainable_vars))
        # Create networks and core TF parts that are shared across setup parts.
        self.actor_tf = actor(normalized_obs0, False, False)
        self.normalized_critic_tf = critic(normalized_obs0, self.actions, False, True)
        self.critic_tf = denormalize(
            tf.clip_by_value(self.normalized_critic_tf, self.return_range[0], self.return_range[1]), self.ret_rms)
        self.normalized_critic_with_actor_tf = critic(normalized_obs0, self.actor_tf, True, True)
        self.critic_with_actor_tf = denormalize(
            tf.clip_by_value(self.normalized_critic_with_actor_tf, self.return_range[0], self.return_range[1]),
            self.ret_rms)
        Q_obs1 = denormalize(target_critic(normalized_obs1, target_actor(normalized_obs1, False, True), False, True), self.ret_rms)
        self.target_Q = self.rewards + (1. - self.terminals1) * gamma * Q_obs1

        self.metrics = []

        self.setup_actor_optimizer()
        self.setup_critic_optimizer()

        self.state_monitor = self.obs0 * tf.constant(1.0, dtype=tf.float32)
        self.action_monitor = self.actions * tf.constant(1.0, dtype=tf.float32)
        self.reward_monitor = self.rewards * tf.constant(1.0, dtype=tf.float32)

        # add metrics
        with tf.name_scope(name + '_training'):
            self.metrics.append(tf.summary.scalar('actor_loss', self.actor_loss))
            self.metrics.append(tf.summary.scalar('critic_loss', self.critic_loss))
            # self.metrics.append(tf.summary.scalar('actor_global_norm', self.actor_global_norm))
            # self.metrics.append(tf.summary.scalar('critic_global_norm', self.critic_global_norm))
            self.metrics.append(tf.summary.scalar('reference_Q_mean', tf.reduce_mean(self.critic_tf)))
            self.metrics.append(tf.summary.scalar('reference_action_mean', tf.reduce_mean(self.actor_tf)))

        with tf.name_scope(name + '_input'):
            # self.metrics.append(tf.summary.scalar('action_avg', tf.reduce_mean(self.actions)))
            # self.metrics.append(tf.summary.scalar('reward_avg', tf.reduce_mean(self.rewards)))
            # self.metrics.append(tf.summary.histogram('obs', self.obs0))
            # self.metrics.append(tf.summary.histogram('action', self.actions))
            self.metrics.append(tf.summary.scalar('action_avg', tf.reduce_mean(self.action_monitor)))
            self.metrics.append(tf.summary.scalar('reward_avg', tf.reduce_mean(self.reward_monitor)))
            self.metrics.append(tf.summary.histogram('obs', self.state_monitor))
            self.metrics.append(tf.summary.histogram('action', self.action_monitor))
        self.merged = tf.summary.merge(self.metrics)

        with tf.control_dependencies([self.actor_train_op, self.critic_train_op]):
            self.setup_target_network_updates()

        # saver and init

        self.saver = tf.train.Saver()
        self.init = tf.global_variables_initializer()

        # for feed
        self.rewards_val = np.random.rand(1, self.rewards.get_shape()[1].value)
        self.terminals1_val = np.random.rand(1, self.terminals1.get_shape()[1].value)
        self.actions_val = np.random.rand(1, self.actions.get_shape()[1].value)
        if len(self.obs1.get_shape().as_list())==2:
            self.obs1_val = np.random.rand(1, self.obs1.get_shape()[1].value)
            self.obs0_val = np.random.rand(1, self.obs0.get_shape()[1].value)
        elif len(self.obs1.get_shape().as_list())==3:
            self.obs1_val = np.random.rand(1, self.obs1.get_shape()[1].value, self.obs1.get_shape()[2].value)
            self.obs0_val = np.random.rand(1, self.obs0.get_shape()[1].value, self.obs0.get_shape()[2].value)
        else:
            print('error. dimension unknow!')

    def setup_target_network_updates(self):
        actor_init_updates, actor_soft_updates = get_target_updates(self.actor.trainable_vars,
                                                                    self.target_actor.trainable_vars, self.tau)
        print('-------len2:',len(self.critic.trainable_vars), len(self.target_critic.trainable_vars))
        critic_init_updates, critic_soft_updates = get_target_updates(self.critic.trainable_vars,
                                                                      self.target_critic.trainable_vars, self.tau)
        self.target_init_updates = [actor_init_updates, critic_init_updates]
        self.target_soft_updates = [actor_soft_updates, critic_soft_updates]

    def setup_actor_optimizer(self):
        print('setting up actor optimizer')
        self.actor_loss = -tf.reduce_mean(self.critic_with_actor_tf)
        actor_shapes = [var.get_shape().as_list() for var in self.actor.trainable_vars]
        actor_nb_params = sum([reduce(lambda x, y: x * y, shape) for shape in actor_shapes])
        print('  actor shapes: {}'.format(actor_shapes))
        print('  actor params: {}'.format(actor_nb_params))
        self.actor_grads = tf.gradients(self.actor_loss, self.actor.trainable_vars)
        if self.clip_norm is not None:
            # self.actor_grads, self.actor_global_norm = tf.clip_by_global_norm(self.actor_grads, self.clip_norm)
            self.actor_grads = [tf.clip_by_norm(x, self.clip_norm) for x in self.actor_grads]
        self.actor_optimizer = tf.train.AdamOptimizer(learning_rate=self.actor_lr, beta1=0.9, beta2=0.999,
                                                      epsilon=1e-08)
        self.actor_train_op = self.actor_optimizer.apply_gradients(zip(self.actor_grads, self.actor.trainable_vars),
                                                                   global_step=self.global_step)


    def setup_critic_optimizer(self):
        print('setting up critic optimizer')
        normalized_critic_target_tf = tf.clip_by_value(normalize(self.target_Q, self.ret_rms), self.return_range[0],
                                                       self.return_range[1])
        self.critic_loss = tf.reduce_mean(tf.square(self.normalized_critic_tf - normalized_critic_target_tf))
        if self.critic_l2_reg > 0.:
            critic_reg_vars = [var for var in self.critic.trainable_vars if
                               'kernel' in var.name and 'output' not in var.name]
            for var in critic_reg_vars:
                print('  regularizing: {}'.format(var.name))
            print('  applying l2 regularization with {}'.format(self.critic_l2_reg))
            critic_reg = tc.layers.apply_regularization(
                tc.layers.l2_regularizer(self.critic_l2_reg),
                weights_list=critic_reg_vars
            )
            self.critic_loss += critic_reg
        critic_shapes = [var.get_shape().as_list() for var in self.critic.trainable_vars]
        critic_nb_params = sum([reduce(lambda x, y: x * y, shape) for shape in critic_shapes])
        print('  critic shapes: {}'.format(critic_shapes))
        print('  critic params: {}'.format(critic_nb_params))
        self.critic_grads = tf.gradients(self.critic_loss, self.critic.trainable_vars)
        if self.clip_norm is not None:
            # self.critic_grads, self.critic_global_norm = tf.clip_by_global_norm(self.critic_grads, self.clip_norm)
            self.critic_grads = [tf.clip_by_norm(x, self.clip_norm) for x in self.critic_grads]
        self.critic_optimizer = tf.train.AdamOptimizer(learning_rate=self.critic_lr, beta1=0.9, beta2=0.999,
                                                       epsilon=1e-08)
        self.critic_train_op = self.critic_optimizer.apply_gradients(zip(self.critic_grads, self.critic.trainable_vars),
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
        self.memory.append(obs0, action, reward, obs1, terminal1)
        if self.normalize_observations:
            self.obs_rms.update(np.array([obs0]))

    def train(self):
        # Get a batch.
        batch = self.memory.sample(batch_size=self.batch_size)

        ops = [self.merged, self.target_soft_updates, self.actor_train_op, self.critic_train_op, self.target_Q]
        self.summary, _, _, _, _ = self.sess.run(ops, feed_dict={
            self.obs1: batch['obs1'],
            self.rewards: batch['rewards'],
            self.terminals1: batch['terminals1'].astype('float32'),
            self.obs0: batch['obs0'],
            self.actions: batch['actions']
        })

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
        self.sess.run(self.target_init_updates, feed_dict={
            self.obs1: self.obs1_val,
            self.rewards: self.rewards_val,
            self.terminals1: self.terminals1_val,
            self.obs0: self.obs0_val,
            self.actions: self.actions_val
        })

    def update_target_net(self):
        self.sess.run(self.target_soft_updates, feed_dict={
            self.obs1: self.obs1_val,
            self.rewards: self.rewards_val,
            self.terminals1: self.terminals1_val,
            self.obs0: self.obs0_val,
            self.actions: self.actions_val
        })

    def reset(self):
        # Reset internal state after an episode is complete.
        if self.action_noise is not None:
            self.action_noise.reset()
