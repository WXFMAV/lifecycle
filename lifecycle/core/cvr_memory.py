import numpy as np
from .memory import RingBuffer


def array_min2d(x):
    x = np.array(x)
    if x.ndim >= 2:
        return x
    return x.reshape(-1, 1)


class CVRMemory(object):
    def __init__(self, limit, observation_shape, action_shape):
        self.limit = limit

        self.observations0 = RingBuffer(limit, shape=observation_shape)
        self.cvr_labels = RingBuffer(limit, shape=(1,))
        self.actions = RingBuffer(limit, shape=action_shape)

    def sample(self, batch_size):
        # Draw such that we always have a proceeding element.
        batch_idxs = np.random.random_integers(self.nb_entries - 2, size=batch_size)

        obs0_batch = self.observations0.get_batch(batch_idxs)
        cvr_label_batch = self.cvr_labels.get_batch(batch_idxs)
        action_batch = self.actions.get_batch(batch_idxs)

        result = {
            'obs0': array_min2d(obs0_batch),
            'cvr_labels': array_min2d(cvr_label_batch),
            'actions': array_min2d(action_batch),
        }
        return result

    def append(self, obs0, action, terminal1, training=True):
        if not training:
            return

        self.observations0.append(obs0)
        self.cvr_labels.append(terminal1)
        self.actions.append(action)

    @property
    def nb_entries(self):
        return len(self.observations0)
