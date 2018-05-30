from copy import copy
from functools import reduce

import tensorflow.contrib as tc
from tensorflow.python.framework.errors_impl import InvalidArgumentError

from util import *


class DDPGFBE(object):
    def __init__(self, name, actor, critic, state_price_model, state_cvr_model,
                 memory, observation_shape, action_shape, price_memory, cvr_memory,
                 param_noise=None, action_noise=None,
                 gamma=0.99, tau=0.001, normalize_returns=False, enable_popart=False, normalize_observations=False,
                 batch_size=128, observation_range=(-5., 5.), action_range=(-1., 1.), return_range=(-np.inf, np.inf),
                 critic_l2_reg=0., actor_lr=1e-4, critic_lr=1e-3, clip_norm=None, reward_scale=1.,
                 ckp_dir=None, global_step=None,
                 obs0=None, obs1=None, terminals1=None, rewards=None, actions=None,
                 price_model_lr=1e-3, cvr_model_lr=1e-3,
                 price_model_batch_size=128, cvr_model_batch_size=256,
                 price_model_y=None, cvr_model_y=None):
        self.name = name
        # Inputs.
        self.obs0 = tf.placeholder(tf.float32, shape=(None,) + observation_shape, name='obs0') if obs0 is None else obs0
        self.obs1 = tf.placeholder(tf.float32, shape=(None,) + observation_shape, name='obs1') if obs1 is None else obs1
        self.terminals1 = tf.placeholder(tf.float32, shape=(None, 1), name='terminals1') if terminals1 is None else terminals1
        self.rewards = tf.placeholder(tf.float32, shape=(None, 1), name='rewards') if rewards is None else rewards
        self.actions = tf.placeholder(tf.float32, shape=(None,) + action_shape, name='actions') if actions is None else actions

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

        # two additional models
        # average transaction amount model (state 2 price model)
        # state 2 cvr model
        self.state_price_model = state_price_model
        self.state_cvr_model = state_cvr_model
        self.price_model_lr = price_model_lr
        self.cvr_model_lr = cvr_model_lr
        self.price_model_training_count = 0
        self.cvr_model_training_count = 0
        self.price_memory = price_memory
        self.cvr_memory = cvr_memory
        self.price_model_batch_size = price_model_batch_size
        self.cvr_model_batch_size = cvr_model_batch_size
        self.train_rl = False
        self.price_model_y = tf.placeholder(tf.float32, shape=(None, 1), name='price_y') \
            if price_model_y is None else price_model_y
        self.cvr_model_y = tf.placeholder(tf.float32, shape=(None, 1), name='cvr_y') \
            if cvr_model_y is None else cvr_model_y
        self.state_price_op = self.state_price_model(self.obs0, self.actions)
        self.state_cvr_op = self.state_cvr_model(self.obs0, self.actions)
        self.def_price_y_value = [0]
        self.def_cvr_y_value = [0]

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

        # Create networks and core TF parts that are shared across setup parts.
        self.actor_tf = actor(normalized_obs0)
        self.normalized_critic_tf = critic(normalized_obs0, self.actions)
        self.critic_tf = denormalize(
            tf.clip_by_value(self.normalized_critic_tf, self.return_range[0], self.return_range[1]), self.ret_rms)
        self.normalized_critic_with_actor_tf = critic(normalized_obs0, self.actor_tf, reuse=True)
        self.critic_with_actor_tf = denormalize(
            tf.clip_by_value(self.normalized_critic_with_actor_tf, self.return_range[0], self.return_range[1]),
            self.ret_rms)
        Q_obs1 = denormalize(target_critic(normalized_obs1, target_actor(normalized_obs1)), self.ret_rms)
        self.target_Q = self.rewards + (1. - self.terminals1) * gamma * Q_obs1

        self.metrics = []
        self.metrics_price_model = []
        self.metrics_cvr_model = []

        self.setup_actor_optimizer()
        self.setup_critic_optimizer()

        # set up the optimizer for state price model and state cvr model
        self.setup_price_model_optimizer()
        self.setup_cvr_model_optimizer()

        # add metrics
        with tf.name_scope(name + '_training'):
            self.metrics.append(tf.summary.scalar('actor_loss', self.actor_loss))
            self.metrics.append(tf.summary.scalar('critic_loss', self.critic_loss))
            # self.metrics.append(tf.summary.scalar('actor_global_norm', self.actor_global_norm))
            # self.metrics.append(tf.summary.scalar('critic_global_norm', self.critic_global_norm))
            self.metrics.append(tf.summary.scalar('reference_Q_mean', tf.reduce_mean(self.critic_tf)))
            self.metrics.append(tf.summary.scalar('reference_action_mean', tf.reduce_mean(self.actor_tf)))
            # add the metric for state price model and state cvr model
            self.metrics_price_model.append(tf.summary.scalar('price_model_loss', self.price_model_loss))
            self.metrics_cvr_model.append(tf.summary.scalar('cvr_model_loss', self.cvr_model_loss))
            self.metrics_price_model.append(tf.summary.scalar('price_mean', tf.reduce_mean(self.state_price_op)))
            self.metrics_cvr_model.append(tf.summary.scalar('cvr_mean', tf.reduce_mean(self.state_cvr_op)))

        with tf.name_scope(name + '_input'):
            self.metrics.append(tf.summary.scalar('action_avg', tf.reduce_mean(self.actions)))
            self.metrics.append(tf.summary.scalar('reward_avg', tf.reduce_mean(self.rewards)))
            self.metrics.append(tf.summary.histogram('obs', self.obs0))
            self.metrics.append(tf.summary.histogram('action', self.actions))
        self.merged = tf.summary.merge(self.metrics)
        self.metrics_price_model_op = tf.summary.merge(self.metrics_price_model)
        self.metrics_cvr_model_op = tf.summary.merge(self.metrics_cvr_model)

        with tf.control_dependencies([self.actor_train_op, self.critic_train_op]):
            self.setup_target_network_updates()

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
        actor_init_updates, actor_soft_updates = get_target_updates(self.actor.trainable_vars,
                                                                    self.target_actor.trainable_vars, self.tau)
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

    def setup_price_model_optimizer(self):
        print("setting up state price model optimizer")
        self.price_model_loss = tf.reduce_mean(tf.square(self.state_price_op - self.price_model_y))
        self.price_model_grads = tf.gradients(self.price_model_loss, self.state_price_model.trainable_vars)
        self.price_model_optimizer = tf.train.AdamOptimizer(learning_rate=self.price_model_lr, beta1=0.9, beta2=0.999,
                                                            epsilon=1e-08)
        self.price_model_train_op = self.price_model_optimizer.apply_gradients(zip(self.price_model_grads,
                                                                                   self.state_price_model.trainable_vars),
                                                                               global_step=self.global_step)

    def setup_cvr_model_optimizer(self):
        print("setting up state cvr model optimizer")
        self.cvr_model_loss = tf.reduce_mean(tf.square(self.state_cvr_op - self.cvr_model_y))
        self.cvr_model_grads = tf.gradients(self.cvr_model_loss, self.state_cvr_model.trainable_vars)
        self.cvr_model_optimizer = tf.train.AdamOptimizer(learning_rate=self.cvr_model_lr, beta1=0.9, beta2=0.999,
                                                            epsilon=1e-08)
        self.cvr_model_train_op = self.cvr_model_optimizer.apply_gradients(zip(self.cvr_model_grads,
                                                                               self.state_cvr_model.trainable_vars),
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
        # reward *= self.reward_scale
        self.memory.append(obs0, action, reward, obs1, terminal1)

        if reward > 0 and not terminal1:
            import random
            if random.random() < 1e-2:
                print("store ipv samples for rl")

        if self.normalize_observations:
            self.obs_rms.update(np.array([obs0]))

    def store_addon_model_samples(self, obs, action, reward, terminal):

        if terminal:
            self.cvr_memory.append(obs, action, 1.0)
            self.price_memory.append(obs, action, reward)
        # we sample pv samples
        elif reward < 1e-5:
            import random
            # if random.random() < 0.3:
            #     self.cvr_memory.append(obs, 0.0)
            if 'android' in self.name:
                if random.random() < 0.9:
                    self.cvr_memory.append(obs, action, 0.0)
            else:
                if random.random() < 0.9:
                    self.cvr_memory.append(obs, action, 0.0)

    def train(self):
        # Get a batch.
        batch = self.memory.sample(batch_size=self.batch_size)

        # first get the average transaction amount and cvr of each state in the batch
        reward_ops = [self.state_price_op, self.state_cvr_op]

        try:
            prices, cvrs = self.sess.run(reward_ops, feed_dict={
                self.obs0: batch['obs0'],
                self.actions: batch['actions'],
                self.rewards: batch['rewards'],
                self.obs1: batch['obs1'],
                self.terminals1: batch['terminals1'].astype('float32'),
                self.price_model_y: [self.def_price_y_value for x in range(self.batch_size)],
                self.cvr_model_y: [self.def_cvr_y_value for x in range(self.batch_size)]
            })
        except InvalidArgumentError:
            print '\nBBBBBB 1', batch['obs0'].shape, self.batch_size

        # print("original reward batch is ")
        # print(batch['rewards'])

        # recompute the immediate reward
        for i in range(0, len(batch['rewards'])):
            if batch['terminals1'][i] or batch['rewards'][i] < 1e-5:
                price = prices[i]
                cvr = cvrs[i]
                batch['rewards'][i] = price * cvr

        # then run the RL operations
        ops = [self.merged, self.target_soft_updates, self.actor_train_op, self.critic_train_op, self.target_Q]

        try:
            self.summary, _, _, _, _ = self.sess.run(ops, feed_dict={
                self.obs0: batch['obs0'],
                self.actions: batch['actions'],
                self.rewards: batch['rewards'],
                self.obs1: batch['obs1'],
                self.terminals1: batch['terminals1'].astype('float32'),
                self.price_model_y: [self.def_price_y_value for x in range(self.batch_size)],
                self.cvr_model_y: [self.def_cvr_y_value for x in range(self.batch_size)]
            })
        except InvalidArgumentError:
            print '\nBBBBBB 2', batch['obs0'].shape, self.batch_size

    def train_price_model(self):
        # print("train price model")
        # Get a batch.
        batch = self.price_memory.sample(batch_size=self.price_model_batch_size)

        ops = [self.price_model_train_op, self.metrics_price_model_op]
        try:
            self.sess.run(ops, feed_dict={
                self.obs0: batch['obs0'],
                self.price_model_y: batch['prices'],
                self.actions: batch['actions'],
                self.obs1: [self.obs1_val[0] for x in range(self.price_model_batch_size)],
                self.rewards: [self.rewards_val[0] for x in range(self.price_model_batch_size)],
                self.terminals1: [self.terminals1_val[0] for x in range(self.price_model_batch_size)],
                self.cvr_model_y: [self.def_cvr_y_value for x in range(self.price_model_batch_size)]
            })
        except InvalidArgumentError:
            print '\nBBBBBB 3', batch['obs0'].shape, self.price_model_batch_size

        # let the training count of two addon models increase
        if not self.train_rl:
            self.price_model_training_count += 1
            print("price model training count is " + str(self.price_model_training_count))
            print("cvr model training count is " + str(self.cvr_model_training_count))
            if self.price_model_training_count > 512 and self.cvr_model_training_count > 4096:
                print("now we can train rl model")
                self.train_rl = True

    def train_cvr_model(self):
        # Get a batch.
        batch = self.cvr_memory.sample(batch_size=self.cvr_model_batch_size)

        ops = [self.cvr_model_train_op, self.metrics_cvr_model_op]
        try:
            self.sess.run(ops, feed_dict={
                self.obs0: batch['obs0'],
                self.cvr_model_y: batch['cvr_labels'],
                self.actions: batch['actions'],
                self.obs1: [self.obs1_val[0] for x in range(self.cvr_model_batch_size)],
                self.rewards: [self.rewards_val[0] for x in range(self.cvr_model_batch_size)],
                self.terminals1: [self.terminals1_val[0] for x in range(self.cvr_model_batch_size)],
                self.price_model_y: [self.def_price_y_value for x in range(self.cvr_model_batch_size)]
            })
        except InvalidArgumentError:
            print '\nBBBBBB 4', batch['obs0'].shape, batch['cvr_labels'].shape

        # let the training count of two addon models increase
        if not self.train_rl:
            self.cvr_model_training_count += 1
            # print("cvr model training count is " + str(self.cvr_model_training_count))
            if self.price_model_training_count > 512 and self.cvr_model_training_count > 4096:
                print("now we can train rl model")
                self.train_rl = True

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

        try:
            self.sess.run(self.target_init_updates, feed_dict={
                self.obs1: self.obs1_val,
                self.rewards: self.rewards_val,
                self.terminals1: self.terminals1_val,
                self.obs0: self.obs0_val,
                self.actions: self.actions_val,
                self.price_model_y: [self.def_price_y_value],
                self.cvr_model_y: [self.def_cvr_y_value]
            })
        except InvalidArgumentError:
            print '\nBBBBBB 5', [self.def_cvr_y_value]

    def update_target_net(self):
        try:
            self.sess.run(self.target_soft_updates, feed_dict={
                self.obs1: self.obs1_val,
                self.rewards: self.rewards_val,
                self.terminals1: self.terminals1_val,
                self.obs0: self.obs0_val,
                self.actions: self.actions_val,
                self.price_model_y: [self.def_price_y_value],
                self.cvr_model_y: [self.def_cvr_y_value]
            })
        except InvalidArgumentError:
            print '\nBBBBBB 6', [self.def_cvr_y_value]

    def reset(self):
        # Reset internal state after an episode is complete.
        if self.action_noise is not None:
            self.action_noise.reset()
