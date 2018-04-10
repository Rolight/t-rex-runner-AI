from game_controller import GameEnv
from models import NNDynamicModel, MPCcontroller
import tensorflow as tf
import time
import json
import pickle
import argparse


def collect_sample_with_mpc_thread(num_samples, output, train_job_name):
    '''
        step 1. make a new dyn_model and load add all parameters
    '''
    sess = tf.Session()
    print('Loading dyn_model from disk...')
    # loading params
    with open('./%s/dyn_model/params.pkl' % train_job_name, 'rb') as f:
        dyn_model_params = pickle.load(f)
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_samples', '-n', type=int)
    parser.add_argument('--output', '-o', type=str)
    parser.add_argument('--train_job_name', '-t', type=str)

    args = parser.parse_args()
    collect_sample_with_mpc_thread(
        args.num_samples, args.output, args.train_job_name)


if __name__ == '__main__':
    main()
