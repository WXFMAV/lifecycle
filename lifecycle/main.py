from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import numpy as np
import math as math
from item import Item
from env import Env
from collections import deque
import argparse
import time
import sys

from core.models import Actor, Critic
from core.memory import Memory
from core.noise import *
from core.ddpg import DDPG
from core.util import *
from ddpg import DDPG2

filepath_param = 'param'

def make_session(num_cpu):
    """Returns a session that will use <num_cpu> CPU's only"""
    tf_config = tf.ConfigProto(
        inter_op_parallelism_threads=num_cpu,
        intra_op_parallelism_threads=num_cpu)
    return tf.Session(config=tf_config)


def single_threaded_session():
    """Returns a session which will only use a single CPU"""
    return make_session(1)

def load_param(filepath_param):
    param = {}
    with open(filepath_param, 'r') as f:
        for line in f:
            k, v = line.strip().replace('\n', '').split(':')
            param[k] = v
    print(param)
    return param 
    assert param is not None
def run(env_id, seed, noise_type, layer_norm, evaluation, **kwargs):
    # Configure things.
    # Create envs.
    np.random.seed(seed) # np.random.seed(int(self.param['random_seed']))
    param = load_param(filepath_param)
#    env = gym.make(env_id)
    env = Env(param)
     
    eval_env = None

    # Parse noise_type
    action_noise = None
    param_noise = None
    nb_actions = env.action_space.shape[-1]
    for current_noise_type in noise_type.split(','):
        current_noise_type = current_noise_type.strip()
        if current_noise_type == 'none':
            pass
        elif 'adaptive-param' in current_noise_type:
            _, stddev = current_noise_type.split('_')
            param_noise = AdaptiveParamNoiseSpec(initial_stddev=float(stddev), desired_action_stddev=float(stddev))
        elif 'normal' in current_noise_type:
            _, stddev = current_noise_type.split('_')
            action_noise = NormalActionNoise(mu=np.zeros(nb_actions), sigma=float(stddev) * np.ones(nb_actions))
        elif 'ou' in current_noise_type:
            _, stddev = current_noise_type.split('_')
            action_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(nb_actions), sigma=float(stddev) * np.ones(nb_actions))
        else:
            raise RuntimeError('unknown noise type "{}"'.format(current_noise_type))
    
    # Configure components.
    memory = Memory(limit=int(25600), action_shape=env.action_space.shape, observation_shape=env.observation_space.shape)

    critic = Critic(net_config=[[64,'relu'],[64,'relu']], layer_norm=layer_norm)
    actor = Actor(net_config=[[64,'relu'],[64,'relu'],[nb_actions,'tanh']], layer_norm=layer_norm)

    # Seed everything to make things reproducible.
    tf.reset_default_graph()
    #env.seed(seed)

    # Disable logging for rank != 0 to avoid noise.
    start_time = time.time()

    train(env=env, eval_env=eval_env, param_noise=param_noise, action_noise=action_noise, actor=actor, critic=critic, memory=memory, **kwargs)
#    train_02(env=env)

    env.close()
    print('total runtime: {}s'.format(time.time() - start_time))

def train_02(env):
    agent = DDPG2(env, env.param)
    state, _, _ = env.reset()
    with tf.Session() as sess:
        step = 0
        train_step = 0
        for step in range(int(env.param['MAX_STEP'])):
            if step < int(env.param['initial_stage_step']):
                action = np.zeros(int(env.param['dim1_action']))
                action[3] = 1.0
            else:
                action = agent.noise_action(state)

            next_state, reward, info = env.step(action)

            print('eps: ', step, ' reward: ', reward)
            if step >= int(env.param['initial_stage_step']) :
                agent.perceive(state, action, reward, next_state, info)
            state = next_state

def train(env, nb_epochs, nb_epoch_cycles, render_eval, reward_scale, render, param_noise, actor, critic,
    normalize_returns, normalize_observations, critic_l2_reg, actor_lr, critic_lr, action_noise,
    popart, gamma, clip_norm, nb_train_steps, nb_rollout_steps, nb_eval_steps, batch_size, memory,
    tau=0.01, eval_env=None, param_noise_adaption_interval=50, **kwargs):

    assert (np.abs(env.action_space.low) == env.action_space.high).all()  # we assume symmetric actions.
    max_action = env.action_space.high
    print('scaling actions by {} before executing in env'.format(max_action))
    global_step = tf.contrib.framework.get_or_create_global_step()
    agent = DDPG('test', actor, critic, memory, env.observation_space.shape, env.action_space.shape,
        gamma=gamma, tau=tau, normalize_returns=normalize_returns, normalize_observations=normalize_observations,
        batch_size=batch_size, action_noise=action_noise, param_noise=param_noise, critic_l2_reg=critic_l2_reg,
        actor_lr=actor_lr, critic_lr=critic_lr, enable_popart=popart, clip_norm=clip_norm,
        reward_scale=reward_scale, ckp_dir=env.param['ckp_dir'], global_step=global_step)
    print('Using agent with the following configuration:')
    print(str(agent.__dict__.items()))

    # Set up logging stuff only for a single worker.

    for v in tf.trainable_variables():
        print(v)
    config = tf.ConfigProto()  
    config.gpu_options.allow_growth=True  
    with tf.Session(config=config) as sess:
        # Prepare everything.
        agent.initialize(sess, init_var=True)
        sess.graph.finalize()

        agent.reset()
        obs, _, _= env.reset()


        step = 0
        train_step = 0 
        for step in range(0, int(env.param['MAX_STEP'])):
            tl = time.time()
            tn = time.time()
            print('[loop]'+str(step)+'[time>start]:'+'%.3f'%(tn - tl))
            tl = tn

            if step < int(env.param['initial_stage_step']):
               action = np.zeros(env.action_space.shape)       
               action[4] = 1.0
            else:
                action, q = agent.pi(obs[0], apply_noise=True, compute_Q=True)
                assert action.shape == env.action_space.shape

            tn = time.time()
            print('[loop]'+str(step)+'[time>env]:'+'%.3f'%(tn - tl))
            tl = tn

            new_obs, reward, info = env.step(action)
            done = 0  # forever can not be done

            tn = time.time()
            print('[loop]' + str(step) + '[time>env]:' + '%.3f'%(tn - tl))
            tl = tn

            print(env.action_space.shape, len(obs), obs[0].shape, step, reward)

            if step > int(env.param['initial_stage_step']):
                for k_new_obs in new_obs:
                   for k_obs in obs:
                       agent.store_transition(k_obs, action, reward, k_new_obs, done)

            tn = time.time()
            print('[loop]' + str(step) + '[time>store]:' + '%.3f'%(tn - tl))
            tl = tn

            obs = new_obs

            if agent.memory.nb_entries > batch_size :
                for t_train in range(nb_train_steps):
                    train_step += 1
                    agent.train()
                    if int(env.param['write_summary']) == 1:
                        agent.write_summary(train_step)
                    #agent.save_model()
            tn = time.time()
            print('[loop]' + str(step) + '[time>train]:' + '%.3f'%(tn - tl))
            tl = tn


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument('--env-id', type=str, default='simulator-v0')
    boolean_flag(parser, 'render-eval', default=False)
    boolean_flag(parser, 'layer-norm', default=True)
    boolean_flag(parser, 'render', default=False)
    boolean_flag(parser, 'normalize-returns', default=False)
    boolean_flag(parser, 'normalize-observations', default=False)
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--critic-l2-reg', type=float, default=1e-2)
    parser.add_argument('--batch-size', type=int, default=64)  # per MPI worker
    parser.add_argument('--actor-lr', type=float, default=1e-4)
    parser.add_argument('--critic-lr', type=float, default=1e-3)
    boolean_flag(parser, 'popart', default=False)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--reward-scale', type=float, default=1.)
    parser.add_argument('--clip-norm', type=float, default=None)
    parser.add_argument('--nb-epochs', type=int, default=5000)  # with default settings, perform 1M steps total
    parser.add_argument('--nb-epoch-cycles', type=int, default=1)
    parser.add_argument('--nb-train-steps', type=int, default=50)  # per epoch cycle and MPI worker
    parser.add_argument('--nb-eval-steps', type=int, default=100)  # per epoch cycle and MPI worker
    parser.add_argument('--nb-rollout-steps', type=int, default=300)  # per epoch cycle and MPI worker
    parser.add_argument('--noise-type', type=str, default='normal_0.2')  # choices are adaptive-param_xx, ou_xx, normal_xx, none
    parser.add_argument('--param-file', type=str, default='param')
    boolean_flag(parser, 'evaluation', default=False)
    return vars(parser.parse_args())


if __name__ == '__main__':

    args = parse_args()
    # Run actual script.
    filepath_param = args['param_file']
    print(filepath_param) 
    run(**args)
