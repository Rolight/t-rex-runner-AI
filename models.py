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

        inputs = self.input_placeholder
        outputs = self.label_placeholder
        # convert reward [-1, 1] to 0, 1
        outputs = (outputs + 1) // 2
        action_inputs = {}
        action_outputs = {}
        for action_id in range(self.env_conf['action_type_size']):
            mask = tf.equal(inputs[:, self.env_conf['observation_dim']], action_id)
            cur_inputs = tf.boolean_mask(inputs, mask)
            cur_inputs = tf.concat(
                [
                    cur_inputs[:, :-self.env_conf['action_dim']],
                    cur_inputs[:, -self.env_conf['action_dim'] + 1:]
                ],
                axis=1
            )
            action_inputs[action_id] = cur_inputs
            action_outputs[action_id] = tf.boolean_mask(outputs, mask)

        self.train_inputs = action_inputs
        self.train_outputs = action_outputs

    def add_prediction_op(self):
        pred = [
            build_mlp(
                input_placeholder=self.train_inputs[action_id],
                output_size=2,
                scope='nndym-%s-action-%d' % (self.name, action_id),
                **self.mlp_params
            )
            for action_id in range(self.env_conf['action_type_size'])
        ]
        return pred

    def add_loss_op(self, pred):
        loss = [
            tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(
                labels=self.train_outputs[action_id], logits=pred[action_id],
            ))
            for action_id in range(self.env_conf['action_type_size'])
        ]
        return loss

    def add_train_op(self, loss):
        train_op = [
            tf.train.AdamOptimizer(
                learning_rate=self.learning_rate[action_id]
            ).minimize(loss[action_id])
            for action_id in range(self.env_conf['action_type_size'])
        ]
        return train_op

    def normalize_obs(self, obs):
        min_obs, max_obs, mean_obs = self.normalization
        obs[:, 1:] = (obs[:, 1:] - mean_obs[1:]) / (max_obs[1:] - min_obs[1:])
        return obs

    def recover_obs(self, obs):
        min_obs, max_obs, mean_obs = self.normalization
        obs[:, 1:] = obs[:, 1:] * (max_obs[1:] - min_obs[1:]) + mean_obs[1:]
        return obs

    def fit(self, data, print_every=10):
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
            if iter_step % print_every == 0:
                print('on iter_step %d, loss = %s' %
                      (iter_step, np.mean(losses, axis=0)))

    def predict(self, states, actions):
        pred = np.empty(dtype=np.float, shape=(len(states), 2))
        states = np.array(states)
        states = self.normalize_obs(states)
        actions = np.array(actions)
        inputs = np.concatenate(
            [states, actions],
            axis=1)
        action_pred = self.sess.run(self.pred, feed_dict={
            self.input_placeholder: inputs
        })
        for action_id in range(self.env_conf['action_type_size']):
            pred[actions[:, 0] == action_id, :] = action_pred[action_id]
        return self._softmax(pred)

    def _softmax(self, vec):
        exp_vec = np.exp(vec)
        return exp_vec / np.sum(exp_vec, axis=1, keepdims=True)


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

        dead_p = rewards[:, -1]
        return actions[np.argmin(dead_p)]
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
