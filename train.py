import datetime
import tensorflow as tf
import pickle
import json
import numpy as np
import os
from pathlib import Path

from models import NNDynamicModel, MPCcontroller
from utils import load_sample_data, DataBuffer, collect_sample_with_mpc
from game_controller import GameEnv

DYNAMIC_MODEL_PARAMS = {
    'kernel_regularizer': tf.contrib.layers.l2_regularizer(scale=1e-5),
    'activation': tf.nn.relu,
    'output_activation': None,
    'learning_rate': 0.003,
    'batch_size': 128,
    'n_layers': 3,
    'size': 128,
    'iterations': 40
}

SAMPLE_DATA_PATH = './sample.json'
DATABUFFER_LIMIT = 50000

TRAIN_PARAMETERS = {
    'sample_size': 30000,
    'collect_data_size': 8000,
    'collect_threads': 4
}


now_datetime = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')


class TrainJob:

    def __init__(self, name=now_datetime):
        self.name = name
        self.sess = tf.Session()
        if not self.can_load():
            self.init_data_buffer()
            self.init_dyn_model()
            self.init_train_params()
        else:
            self.load()
            self.load_data_buffer()
            self.load_dyn_model()
        self.init_mpc_controller()

    def init_train_params(self):
        self.params = TRAIN_PARAMETERS

    def init_data_buffer(self):
        '''
            Init a new data buffer.
            Preload random sample from sample.json.
            If you don't have this file, you need to run generate_random_sample to create one
        '''
        init_data = load_sample_data(SAMPLE_DATA_PATH)
        self.db = DataBuffer(init_data, DATABUFFER_LIMIT)

    def init_dyn_model(self):
        '''
            Init a new dyn model with DYNAMIC_MODEL_PARAMS
        '''
        self.dyn_model_params = DYNAMIC_MODEL_PARAMS
        self.dyn_model = NNDynamicModel(
            name=self.name, sess=self.sess, **self.dyn_model_params)

    def init_mpc_controller(self):
        '''
            Init mpc controller
        '''
        self.mpc = MPCcontroller(dyn_model=self.dyn_model)

    def save_dyn_model(self):
        print('Saving dyn_model to disk...')
        # create path
        path = Path('./%s/dyn_model/' % self.name)
        path.mkdir(parents=True, exist_ok=True)
        # saving model
        saver = tf.train.Saver()
        save_path = saver.save(
            self.sess, './%s/dyn_model/model.ckpt' % self.name)
        print('Model has saved in %s' % save_path)
        # saving params
        with open('./%s/dyn_model/params.pkl' % self.name, 'wb') as f:
            pickle.dump(self.dyn_model_params, f)
        print('Model parameter has saved in %s' %
              ('./%s/dyn_model/params.pkl' % self.name))

    def save_dataBuffer(self):
        print('Saving databuffer to disk...')
        path = Path('./%s/' % self.name)
        path.mkdir(parents=True, exist_ok=True)
        with open('./%s/data_buffer.pkl', 'wb') as f:
            pickle.dump(self.db, f)
        print('DataBuffer has saved to %s' %
              ('./%s/data_buffer.pkl' % self.name))

    def save(self):
        print('Savning training parameters to disk...')
        path = Path('./%s/' % self.name)
        path.mkdir(parents=True, exist_ok=True)
        self.params['name'] = self.name
        with open('./%s/parameters.pkl', 'wb') as f:
            pickle.dump(self.params, f)
        print('Training parameters has saved to %s' %
              ('./%s/parameters.pkl' % self.name))

    def can_load(self):
        path = Path('./%s/' % self.name)
        return path.exists()

    def load_data_buffer(self):
        print('Loading databuffer from disk...')
        with open('./%s/data_buffer.pkl', 'rb') as f:
            self.db = pickle.load(f)
        print('DataBuffer has loaded from %s' %
              ('./%s/data_buffer.pkl' % self.name))

    def load_dyn_model(self):
        print('Loading dyn_model from disk...')
        # loading params
        with open('./%s/dyn_model/params.pkl' % self.name, 'rb') as f:
            self.dyn_model_params = pickle.dump(f)
        print('Model parameter has loaded from %s' %
              ('./%s/dyn_model/params.pkl' % self.name))
        # building graph
        self.dyn_model = NNDynamicModel(
            sess=self.sess, **self.dyn_model_params)
        # restore model
        saver = tf.train.Saver()
        saver.restore(self.sess, './%s/dyn_model/model.ckpt' % self.name)

    def load(self):
        print('Loading training parameters from disk...')
        with open('./%s/parameters.pkl', 'rb') as f:
            self.params = pickle.load(f)
        print('Training parameters has loaded from %s' %
              ('./%s/parameters.pkl' % self.name))

    def train_loop(self):
        '''
            step 1. Train dynamic model with sample data.
                    After training, save model to disk.
        '''
        data = self.db.sample(sample_size=self.params['sample_size'])
        self.dyn_model.fit(data)
        self.save_dyn_model()
        '''
            step 2. Run game and using MPC to collect new data.
                    Add new data to DataBuffer.
                    Save DataBuffer to disk
        '''
        data = collect_sample_with_mpc(
            num_samples=self.params['collect_data_size'],
            num_threads=self.params['collect_threads'])
        self.db.add_data(data)
        self.save_dataBuffer()

    def train(self):
        if not self.dyn_model.initialized:
            initialize = tf.global_variables_initializer()
            self.sess.run(initialize)
        while True:
            self.train_loop()
            self.save()