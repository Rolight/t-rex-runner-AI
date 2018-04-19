import tensorflow as tf
import time
import argparse

from train import TrainJob
from game_controller import GameEnv
from models import HeuristicMPCcontroller


def run(brain_name):
    train_job = TrainJob(name=brain_name)
    env = GameEnv()
    mpc = HeuristicMPCcontroller(
        dyn_model=train_job.dyn_model, sample_size=20, min_tolerance=0.95)
    # mpc = train_job.mpc
    env.start()
    last_obs = env.get_observation()
    obs = None
    while True:
        while last_obs is None:
            time.sleep(env.interval_time)
            last_obs = env.get_observation()
        if env.status == 'CRASHED':
            break
        # using mpc controller to get action
        action = mpc.get_action(last_obs)
        obs, done, reward = env.perform_action(action)
        if done:
            env.restart()
            last_obs = None
        else:
            last_obs = obs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--brain_name', '-n', type=str)

    args = parser.parse_args()
    run(args.brain_name)


if __name__ == '__main__':
    main()
