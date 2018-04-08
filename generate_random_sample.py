import argparse
import threading
import time
import json
import os

from multiprocessing import Process

from game_controller import GameEnv


def collect_sample(num_samples, output, pid):
    print('Process %d start' % pid)
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
        action = env.sample_action()
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
            print('Process %d have collected %d samples.' %
                  (pid, sample_count))
    with open('%s-%d' % (output, pid), 'w') as f:
        f.write(json.dumps(samples))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_threads', '-t', type=int, default=4)
    parser.add_argument('--output', '-o', type=str, default='sample.json')
    parser.add_argument('--num_samples', '-n', type=int, default=10000)

    args = parser.parse_args()
    num_samples_each_thread = args.num_samples // args.num_threads
    processes = []
    for pid in range(args.num_threads):
        if pid == args.num_threads - 1:
            num_samples_each_thread += args.num_samples % args.num_threads
        p = Process(target=collect_sample, args=(
            num_samples_each_thread, args.output, pid))
        processes.append(p)

    for p in processes:
        p.start()

    for p in processes:
        p.join()

    print('all process finished, writing data to output file...')
    all_samples = []
    for pid in range(args.num_threads):
        with open('%s-%d' % (args.output, pid), 'r') as f:
            cur_samples = json.loads(f.read())
            all_samples += cur_samples
        os.remove('%s-%d' % (args.output, pid))
    with open(args.output, 'w') as f:
        f.write(json.dumps(all_samples))


if __name__ == '__main__':
    main()
