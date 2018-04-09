import tensorflow as tf
import numpy as np

from utils import build_mlp

ENV_CONF = {
    'action_dim': 2,
    'action_type_size': 3,
    'action_hold_time_limit': 1,
    'observation_dim': 5
}


class NNDynamicModel:

    def __init__(self):
        pass

    def predict(self, states, actions):
        nxt_states, reward = None, None
        return nxt_states, rewards


class MPCcontroller:

    def __init__(self, dyn_model, sample_size=20, env_conf=ENV_CONF):
        self.dyn_model = dyn_model
        self.sample_size = sample_size
        self.env_conf = env_conf

    def get_random_actions(self, action_type):
        actions = np.empty(
            shape=(self.sample_size, self.env_conf['action_dim']))
        actions[:, 0] = action_type
        actions[:, 1] = np.random.uniform(
            low=0, high=self.env_conf['action_hold_time_limit'],
            size=(self.sample_size,))
        return actions

    def get_action(self, state):
        actions = np.empty(
            shape=(self.sample_size * self.env_conf['action_type_size'],
                   self.env_conf['action_dim']))
        observations = np.empty(
            shape=(self.sample_size * self.env_conf['action_type_size'],
                   self.env_conf['action_dim']))
        observations[:, :] = state
        _, rewards = self.dyn_model.predict(observations, actions)
        '''
        select mpc optimal action
        for rewards, we perfer which reward is > 0
        for action_type and hold_time, we perfer samller one
        '''
        if np.sum(rewards > 0) == 0:
            # if game will fail no matter which action we take
            # return a random action
            return actions[np.random.randint(0, actions.shape[0] + 1)]

        actions = actions[rewards > 0, :]
        actions.sort(axis=0)
        return actions[0]


class TexRunnerBrain:

    def __init__(self, params):
        self.params = params
