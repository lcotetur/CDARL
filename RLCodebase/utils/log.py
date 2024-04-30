import numpy as np
import os
from collections import deque
import json
import matplotlib.pyplot as plt

class LoggerTensorboard:
    def __init__(self, writer, num_echo_episodes, episodes_avg_window = -1):
        self.writer = writer
        assert(num_echo_episodes >= 0)
        self.last_rewards = deque(maxlen=num_echo_episodes)
        self.episodes_avg_window = episodes_avg_window
        self.window_rewards = []
        self.window_end = self.episodes_avg_window

    def save_episodic_return(self, infos, done_steps):
        for (i, info) in enumerate(infos):
            if info['episodic_return'] is not None:
                global_step = done_steps + i
                # save each episode return or save average episodic return in windows
                if self.episodes_avg_window == -1:
                    if self.writer is not None:
                        self.writer.add_scalar('episodic_return', info['episodic_return'], done_steps+i)
                else:
                    if global_step < self.window_end:
                        self.window_rewards.append(info['episodic_return']) 
                    else:
                        if self.writer is not None and len(self.window_rewards) != 0:
                            self.writer.add_scalar('episodic_return', np.mean(self.window_rewards), self.window_end)
                        self.window_rewards = [info['episodic_return']]
                        self.window_end = (global_step // self.episodes_avg_window + 1) * self.episodes_avg_window
                self.last_rewards.append(info['episodic_return'])

    def print_last_rewards(self):
        avg_return = 'None' if len(self.last_rewards) == 0 else str(np.mean(self.last_rewards))
        print ('Average episodic return of %d episodes is: %s' % (len(self.last_rewards), avg_return))


    def add_scalar(self, tags, values, step):
        if self.writer is None:
            return
        for (i, tag) in enumerate(tags):
            if values[i] is not None:
                self.writer.add_scalar(tag, values[i], step if isinstance(step, int) else step[i])


    def close(self):
        self.writer.close()

class Logger:
    def __init__(self, save_path, num_echo_episodes, episodes_avg_window = -1):
        self.writer = {'done step': [], 'episode': [], 'episodic_return': [], 'running average': [], 'eval_returns': [], 'action_loss': [], 'value_loss': [], 'entropy': []}
        self.save_path = save_path
        assert(num_echo_episodes >= 0)
        self.last_rewards = deque(maxlen=num_echo_episodes)
        self.episodes_avg_window = episodes_avg_window
        self.window_rewards = []
        self.window_end = self.episodes_avg_window

    def save_episodic_return(self, infos, done_steps):
        for (i, info) in enumerate(infos):
            if info['episodic_return'] is not None:
                global_step = done_steps + i
                # save each episode return or save average episodic return in windows
                if self.episodes_avg_window == -1:
                    if self.writer is not None:
                        running_average = 0.99 * np.mean(self.writer['episodic_return']) + 0.01 * np.mean(info['episodic_return'])
                        self.writer['episodic_return'].append(info['episodic_return'])
                        self.writer['running average'].append(running_average)
                        self.writer['done step'].append(done_steps+i)
                        self.writer['episode'].append(len(self.writer['episode']) + 1)
                else:
                    if global_step < self.window_end:
                        self.window_rewards.append(info['episodic_return']) 
                    else:
                        if self.writer is not None and len(self.window_rewards) != 0:
                            running_average = 0.99 * np.mean(self.writer['episodic_return']) + 0.01 * np.mean(self.window_rewards)
                            self.writer['episodic_return'].append(np.mean(self.window_rewards))
                            self.writer['running average'].append(running_average)
                            self.writer['done step'].append(self.window_end)
                            self.writer['episode'].append(len(self.writer['episode']) + 1)
                        self.window_rewards = [info['episodic_return']]
                        self.window_end = (global_step // self.episodes_avg_window + 1) * self.episodes_avg_window
                self.last_rewards.append(info['episodic_return'])

    def print_last_rewards(self):
        avg_return = 'None' if len(self.last_rewards) == 0 else str(np.mean(self.last_rewards))
        print ('Average episodic return of %d episodes is: %s' % (len(self.last_rewards), avg_return))

    def add_scalar(self, tags, values, step):
        if self.writer is None:
            return
        for (i, tag) in enumerate(tags):
            if values[i] is not None:
                self.writer[tag].append(values[i])

    def save_logs(self):
        with open(os.path.join(self.save_path, "logs.json"), 'w') as f:
            json.dump(self.writer, f, indent=2)

    def save_plot(self):
        """ Generate plot according to log 
        """
        fig, ax = plt.subplots()
        ax.plot(self.writer['episode'],self.writer['running average'], label="Moving averaged episode reward")
        ax.set_xlabel('episode')
        ax.set_ylabel('running average')
        fig.suptitle("Moving averaged episode reward")
        fig.savefig(os.path.join(self.save_path, 'running_average.png'))

class MultiDeque:
    def __init__(self, tags = None):
        self.tags = tags
        if self.tags is not None:
            self.queues = [deque() for _ in range(len(tags))]
        else:
            self.queues = None

    def add(self, data):
        if self.queues is None:
            self.queues = [deque() for _ in range(len(data))]
        for i in range(len(data)):
            self.queues[i].append(data[i])

    def clear(self):
        for q in self.queues:
            q.clear()

    def set_tags(self, tags):
        self.tags = tags

    def return_summary(self):
        values  = [np.mean(q) for q in self.queues]
        self.clear()
        return self.tags, values

    # could only write if self.tags exists
    def write(self, writer, step):
        if writer is None:
            return
        assert(self.tags is not None)
        result = [np.mean(q) for q in self.queues]
        for i, r in enumerate(result):
            writer.add_scalar(self.tags[i], result[i], step)