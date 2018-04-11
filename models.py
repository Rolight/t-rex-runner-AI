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

    def __init__(self,
                 name,
                 n_layers,
                 size,
                 activation,
                 l2_regularizer_scale,
                 normalization,
                 batch_size,
                 iterations,
                 learning_rate,
                 sess,
                 output_activation,
                 env_conf=ENV_CONF
                 ):
        # base parameters
        self.batch_size = batch_size
        self.sess = sess
        self.normalization = normalization
        self.iter = iterations
        self.name = name
        self.env_conf = env_conf

        kernel_regularizer = tf.contrib.layers.l2_regularizer(
            scale=l2_regularizer_scale)
        activation = getattr(tf.nn, activation)

        self.mlp_params = {
            'scope': 'nndym-%s' % name,
            'n_layers': n_layers,
            'size': size,
            'activation': activation,
            'output_activation': output_activation,
            'kernel_regularizer': kernel_regularizer
        }
        self.learning_rate = learning_rate
        self.initialized = False

        self.build_graph()

    def build_graph(self):
        self.add_placeholder()
        self.pred = self.add_prediction_op()
        self.loss = self.add_loss_op(self.pred)
        self.train_op = self.add_train_op(self.loss)

    def add_placeholder(self):
        # input [observation, action]
        self.input_placeholder = tf.placeholder(
            dtype=tf.float32,
            shape=[None, self.env_conf['observation_dim'] +
                   self.env_conf['action_dim']]
        )

        # label [reward]
        self.label_placeholder = tf.placeholder(
            dtype=tf.int32,
            shape=[None]
        )

    def add_prediction_op(self):
        inputs = self.input_placeholder
        pred = build_mlp(
            input_placeholder=inputs,
            output_size=2,
            **self.mlp_params
        )
        return pred

    def add_loss_op(self, pred):
        labels = (self.label_placeholder + 1) // 2
        loss = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(
            labels=labels, logits=pred
        ))
        return loss

    def add_train_op(self, loss):
        train_op = tf.train.AdamOptimizer(
            learning_rate=self.learning_rate
        ).minimize(loss)
        return train_op

    def normalize_obs(self, obs):
        min_obs, max_obs, mean_obs = self.normalization
        obs[:, 1:] = (obs[:, 1:] - mean_obs[1:]) / (max_obs[1:] - min_obs[1:])
        return obs

    def recover_obs(self, obs):
        min_obs, max_obs, mean_obs = self.normalization
        obs[:, 1:] = obs[:, 1:] * (max_obs[1:] - min_obs[1:]) + mean_obs[1:]
        return obs

    def fit(self, data):
        """
        data format: [[obs, action, next_obs, reward]]
        """
        inputs_data = np.array([d[0] + d[1] for d in data])
        obs_len = len(data[0][0])
        inputs_data[:, :obs_len] = self.normalize_obs(inputs_data[:, :obs_len])
        outputs_data = np.array([d[-1] for d in data])
        train_indicies = np.arange(len(data))
        for iter_step in range(self.iter):
            np.random.shuffle(train_indicies)
            losses = []
            for i in range(len(train_indicies) // self.batch_size):
                start_id = i * self.batch_size
                train_ids = train_indicies[start_id: start_id+self.batch_size]
                input_batch = inputs_data[train_ids]
                output_batch = outputs_data[train_ids]

                loss, _ = self.sess.run([self.loss, self.train_op], feed_dict={
                    self.input_placeholder: input_batch,
                    self.label_placeholder: output_batch,
                })
                losses.append(loss)
            print('on iter_step %d, loss = %0.7f, max_loss = %0.7f, min_loss = %0.7f' %
                  (iter_step, np.mean(losses), np.max(losses), np.min(losses)))

    def predict(self, states, actions):
        states = np.array(states)
        states = self.normalize_obs(states)
        inputs = np.concatenate([states, actions], axis=1)
        pred = self.sess.run(tf.nn.softmax(self.pred), feed_dict={
            self.input_placeholder: inputs
        })
        return pred


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
        actions = np.concatenate([
            self.get_random_actions(action_type)
            for action_type in range(self.env_conf['action_type_size'])
        ], axis=0)
        observations = np.empty(
            shape=(self.sample_size * self.env_conf['action_type_size'],
                   self.env_conf['observation_dim']))
        observations[:, :] = state
        rewards = self.dyn_model.predict(observations, actions)
        return actions[np.argmax(rewards)]
        '''
        select mpc optimal action
        for rewards, we perfer which reward is > 0
        for action_type and hold_time, we perfer samller one
        '''
        '''
        if np.sum(rewards > 0) == 0:
            # if game will fail no matter which action we take
            # return a random action
            return actions[np.random.randint(0, actions.shape[0] + 1)]

        actions = actions[rewards > 0, :]
        actions.sort(axis=0)
        return actions[0]
        '''
