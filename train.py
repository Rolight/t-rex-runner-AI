import datetime
import tensorflow as tf
import pickle
import json
import numpy as np

from models import NNDynamicModel, MPCcontroller
from utils import load_sample_data, DataBuffer

DYNMAMIC_MODEL_PARAMS = {
    'regularizer': tf.contrib.layers.l2_regularizer(scale=1e-5),
    'activation': tf.nn.relu,
    'output_activation': None,
    'learning_rate': 0.003,
    'batch_size': 128,
    'n_layers': 3,
    'size': 128,
    'iterations': 40
}


now_datetime = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')


class TrainJob:
