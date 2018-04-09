import tensorflow as tf
import json
import numpy as np
import pickle
import os
import time

from multiprocessing import Process


def build_mlp(input_placeholder,
              output_size,
              scope,
              n_layers=2,
              size=500,
              activation=tf.tanh,
              output_activation=None,
              kernel_regularizer=None
              ):
    out = input_placeholder
    with tf.variable_scope(scope):
        for _ in range(n_layers):
            out = tf.layers.dense(
                out, size, activation=activation, kernel_regularizer=kernel_regularizer)
        out = tf.layers.dense(out, output_size, activation=output_activation)
    return out


def load_sample_data(path):
    with open(path, 'r') as f:
        return json.loads(f.read())


class DataBuffer:

    def __init__(self, init_data, limit):
        # YOU SHOULDN'T MODIFY BUFFER LIMIT AT ANY TIME
        self._limit = limit
        self._data = np.empty(shape=(self._limit, ), dtype=np.object)
        self._tail = 0
        self.add_data(init_data)

    def sample(self, sample_size=None):
        indices = np.arange(self._tail)
        np.random.shuffle(indices)
        return self._data[indices[:sample_size]]

    def add_data(self, new_data):
        new_data = np.array(new_data)
        if new_data.shape[0] > self._limit:
            print('Fatal Error! Data is too large to add to data buffer')
            return
        if self._tail + new_data.shape[0] > self._limit:
            slots_need_to_delete = new_data.shape[0] - \
                (self._limit - self._tail)
            remain_slots = self._tail - slots_need_to_delete
            self._data[:remain_slots] = self._data[slots_need_to_delete:self._tail]
            self._tail -= slots_need_to_delete

        self._data[self._tail:self._tail + new_data.shape[0]] = new_data
        self._tail += new_data.shape[0]

    def __str__(self):
        return 'DataBuff with limit %d: ' % self._limit + str(self._data)

    def __repr__(self):
        return str(self)


def _collect_sample_with_mpc_thread(num_samples, output, train_job_name):
    from game_controller import GameEnv
    from models import NNDynamicModel, MPCcontroller
    import tensorflow as tf

    '''
        step 1. make a new dyn_model and load add all parameters
    '''
    sess = tf.Session()
    print('Loading dyn_model from disk...')
    # loading params
    with open('./%s/dyn_model/params.pkl' % train_job_name, 'rb') as f:
        dyn_model_params = pickle.dump(f)
    print('Model parameter has loaded from %s' %
          ('./%s/dyn_model/params.pkl' % train_job_name))
    # building graph
    dyn_model = NNDynamicModel(
        sess=sess, **dyn_model_params)
    # restore model
    saver = tf.train.Saver()
    saver.restore(sess, './%s/dyn_model/model.ckpt' % train_job_name)

    '''
        step 2. init a new mpc controller with dyn_model
    '''

    mpc = MPCcontroller(dyn_model=dyn_model)

    '''
        step 3. create a new env and collect samples
    '''
    print_every = 100
    env = GameEnv()
    sample_count = 0
    env.start()
    last_obs = env.get_observation()
    obs = None
    samples = []
    while sample_count < num_samples:
        while last_obs is None:
            time.sleep(env.interval_time)
            last_obs = env.get_observation()
        # using mpc controller to get action
        action = mpc.get_action(last_obs)
        obs, done, reward = env.perform_action(action)
        if obs is None:
            # Taken last operation make game almost failure, we just mark it was done
            # and mark last sample's reward to be -1
            done = True
            samples[-1][-1] = -1
        else:
            samples.append(
                [last_obs.tolist(), action.tolist(), obs.tolist(), reward])
            sample_count += 1
        if done:
            env.restart()
            last_obs = None
        else:
            last_obs = obs

        if sample_count % print_every == 0:
            print('Process %s have collected %d samples.' %
                  (output, sample_count))
    with open(output, 'w') as f:
        f.write(json.dumps(samples))


def collect_sample_with_mpc(num_samples, num_threads, train_job_name):
    num_samples_each_thread = num_samples // num_threads
    processes = []
    for pid in range(num_threads):
        if pid == num_threads - 1:
            num_samples_each_thread += num_samples % num_threads
        p = Process(target=_collect_sample_with_mpc_thread, args=(
            num_samples_each_thread, 'sample_process_%d' % pid, train_job_name))
        processes.append(p)

    for p in processes:
        p.start()

    for p in processes:
        p.join()

    print('all process finished, collecting data from output file...')
    all_samples = []
    for pid in range(num_threads):
        with open('sample_process_%d' % pid, 'r') as f:
            cur_samples = json.loads(f.read())
            all_samples += cur_samples
        os.remove('sample_process_%d' % pid)
    return all_samples
