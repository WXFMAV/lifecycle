import tensorflow as tf
import numpy as np

def boolean_flag(parser, name, default=False, help=None):
    """Add a boolean flag to argparse parser.

    Parameters
    ----------
    parser: argparse.Parser
        parser to add the flag to
    name: str
        --<name> will enable the flag, while --no-<name> will disable it
    default: bool or None
        default value of the flag
    help: str
        help string for the flag
    """
    dest = name.replace('-', '_')
    parser.add_argument("--" + name, action="store_true", default=default, dest=dest, help=help)
    parser.add_argument("--no-" + name, action="store_false", dest=dest)


def reduce_var(x, axis=None, keepdims=False):
    m = tf.reduce_mean(x, axis=axis, keep_dims=True)
    devs_squared = tf.square(x - m)
    return tf.reduce_mean(devs_squared, axis=axis, keep_dims=keepdims)


def reduce_std(x, axis=None, keepdims=False):
    return tf.sqrt(reduce_var(x, axis=axis, keepdims=keepdims))


def normalize(x, stats):
    if stats is None:
        return x
    return (x - stats.mean) / stats.std


def denormalize(x, stats):
    if stats is None:
        return x
    return x * stats.std + stats.mean


def get_target_updates(vars, target_vars, tau):
    print('setting up target updates ...')
    soft_updates = []
    init_updates = []
    assert len(vars) == len(target_vars)
    for var, target_var in zip(vars, target_vars):
        print('  {} <- {}'.format(target_var.name, var.name))
        init_updates.append(tf.assign(target_var, var))
        soft_updates.append(tf.assign(target_var, (1. - tau) * target_var + tau * var))
    assert len(init_updates) == len(vars)
    assert len(soft_updates) == len(vars)
    return tf.group(*init_updates), tf.group(*soft_updates)


def process_data(data):
    bucket = data[2]
    apptype = data[3]

    obs = np.asarray([map(float, x.split(',')) for x in data[4]])
    action = np.asarray([map(float, x.split(',')) for x in data[5]])
    reward = np.asarray([float(x) for x in data[6]])
    obs_p = np.asarray([obs[i] if data[7][i] == 'ST' else map(float, data[7][i].split(',')) for i in range(len(data[7]))])
    terminal = np.asarray([x == 'ST' for x in data[7]])
    return bucket, apptype, obs, action, reward, obs_p, terminal


def reprocess_data(buckets, apps, obs, actions, rewards, next_obs, done):
    new_buckets = []
    new_apps = []
    new_obs = []
    new_actions = []
    new_rewards = []
    new_next_obs = []
    new_done = []

    # get the user top, use high and user power from obs
    user_top_true_index = 1
    user_high_true_index = 3
    user_power_high_index = 6

    remove_flags = []
    for i in range(0, len(obs)):
        bucket = buckets[i]
        state = obs[i]
        is_terminal = done[i]
        # buckets 1 and 2
        if bucket == '1' or bucket == '2' or bucket == '3':
            # print "bucket is 1 or 2"
            remove_flags.append(False)
        # we only keep the pay samples for advanced users
        elif state[user_top_true_index] > 0 or \
            state[user_high_true_index] > 0 or \
            state[user_power_high_index] > 0:
            if is_terminal:
                remove_flags.append(False)
            else:
                remove_flags.append(True)
                # print "remove pv sample for advanced users"
                # print(state)
        # we keep all samples for low users
        else:
            remove_flags.append(False)

    for i in range(0, len(obs)):
        if not remove_flags[i]:
            new_buckets.append(buckets[i])
            new_apps.append(apps[i])
            new_obs.append(obs[i])
            new_actions.append(actions[i])
            new_rewards.append(rewards[i])
            new_next_obs.append(next_obs[i])
            new_done.append(done[i])

    new_buckets = np.asarray(new_buckets)
    new_apps = np.asarray(new_apps)
    new_obs = np.asarray(new_obs)
    new_actions = np.asarray(new_actions)
    new_rewards = np.asarray(new_rewards)
    new_next_obs = np.asarray(new_next_obs)
    new_done = np.asarray(new_done)
    return new_buckets, new_apps, new_obs, new_actions, new_rewards, new_next_obs, new_done


class OnlineMeanVariance(object):
    """
    Welford's algorithm computes the sample variance incrementally.
    """

    def __init__(self, iterable=None, ddof=1):
        self.ddof, self.n, self.mean, self.M2 = ddof, 0, 0.0, 0.0
        if iterable is not None:
            for datum in iterable:
                self.include(datum)

    def include(self, datum):
        self.n += 1
        self.delta = datum - self.mean
        self.mean += self.delta / self.n
        self.M2 += self.delta * (datum - self.mean)

    @property
    def variance(self):
        return self.M2 / (self.n - self.ddof)

    @property
    def std(self):
        return np.sqrt(self.variance)
