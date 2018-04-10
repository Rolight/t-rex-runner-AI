import tensorflow as tf
import json
import numpy as np
import pickle
import os
import time
import subprocess


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

    def __init__(self, init_data, data_dim, limit):
        # YOU SHOULDN'T MODIFY BUFFER LIMIT AT ANY TIME
        self._limit = limit
        self._data = np.empty(shape=(self._limit, data_dim), dtype=np.object)
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


def collect_sample_with_mpc(num_samples, num_threads, train_job_name):
    num_samples_each_thread = num_samples // num_threads
    processes = []
    for pid in range(num_threads):
        if pid == num_threads - 1:
            num_samples_each_thread += num_samples % num_threads
        p = subprocess.Popen(
            ['python', 'generate_sample_with_mpc.py',
             '--num_samples', str(num_samples_each_thread),
             '--output', 'sample_process_%d' % pid,
             '--train_job_name', train_job_name])
        processes.append(p)

    for p in processes:
        p.wait()

    print('all process finished, collecting data from output file...')
    all_samples = []
    for pid in range(num_threads):
        with open('sample_process_%d' % pid, 'r') as f:
            cur_samples = json.loads(f.read())
            all_samples += cur_samples
        os.remove('sample_process_%d' % pid)
    return all_samples
