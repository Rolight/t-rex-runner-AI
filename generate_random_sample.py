import argparse
import threading
import time
import json

from multiprocessing import Process, Queue

from game_controller import GameEnv


def collect_sample(num_samples, q, pid):
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
        obs, done, reward = env.perform_action(action, verbose=True)
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
    q.put(samples)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_threads', '-t', type=int, default=4)
    parser.add_argument('--output', '-o', type=str, default='sample.json')
    parser.add_argument('--num_samples', '-n', type=int, default=10000)

    args = parser.parse_args()
    num_samples_each_thread = args.num_samples // args.num_threads
    processes = []
    q = Queue()
    for pid in range(args.num_threads):
        if pid == args.num_threads - 1:
            num_samples_each_thread += args.num_samples % args.num_threads
        p = Process(target=collect_sample, args=(
            num_samples_each_thread, q, pid))
        processes.append(p)

    for p in processes:
        p.start()

    for p in processes:
        p.join()

    all_samples = []
    for p in processes:
        all_samples += q.get()
    with open(args.output, 'w') as f:
        f.write(json.dumps(all_samples))


if __name__ == '__main__':
    main()
