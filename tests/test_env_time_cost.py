import gymnasium as gym
from gymnasium.wrappers import HumanRendering
import time
import envs  # noqa
from envs.wrappers.pretext_wrapper import PretextWrapper
from envs.wrappers.raster_wrapper import RasterWrapper

render = False
env = gym.make(
    'UavEnv-v6',
    dimensions=(800, 800),
    use_lidar=False,
    dynamic_obstacles=1,
    render_mode='rgb_array' if render else None,
)
env = PretextWrapper(env, pretext_dir=None, device='cpu')
env = RasterWrapper(env)
if render:
    env = HumanRendering(env)
obs, _ = env.reset()
if render:
    env.render()
total = 500
i = 0
count = []

while True:
    if i == total:
        break
    action = env.action_space.sample()
    # action = 2 * 1
    tic = time.time()
    obs, reward, done, truncated, _ = env.step(action)
    toc = time.time()
    print(f"{i} / {total} | Elapsed time: {100 * (toc - tic)}ms")
    if i > 0:
        count.append(toc - tic)
    i += 1
    # print(reward)
    if render:
        env.render()
    if done:
        obs, _ = env.reset()
        if render:
            env.render()
env.close()
avg_time = sum(count) / len(count)

print(f'Average time: {avg_time * 100:.4}ms')
