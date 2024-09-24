from collections import deque

import numpy as np
from tqdm import trange

from uav_envs import UAVEnvironment

len_dataset = 30000
seq_len = 20

dim = (800, 800)
env = UAVEnvironment(dim, dynamic_obstacles=1)

dtype = [('x', np.float32, (seq_len, 2)), ('len', np.long)]
dataset = np.memmap('data/pretext_dataset.npy', mode='r+', shape=(len_dataset,), dtype=dtype)
idx = 0

done = True
for i in trange(len_dataset):
    if done:
        obs, info = env.reset()
        done = False
        memory = deque(maxlen=seq_len)
    else:
        obs, reward, done, _, info = env.step((0, 0))
        data = env.d_obstacles[0].x / dim[0], env.d_obstacles[0].y / dim[1]
    memory.append((env.d_obstacles[0].x / dim[0], env.d_obstacles[0].y / dim[1]))
    dataset['x'][i] = np.array(list(memory) + (seq_len - len(memory)) * [(0., 0.)])
    dataset['len'][i] = len(memory)
env.close()
