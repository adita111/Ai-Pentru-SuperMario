import gym
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace

env = gym_super_mario_bros.make(
    "SuperMarioBros-v1",
    apply_api_compatibility=True,
    render_mode="human"
)
env = JoypadSpace(env, SIMPLE_MOVEMENT)

obs, info = env.reset()
done = False
truncated = False

for step in range(500000):
    if done or truncated:
        obs, info = env.reset()

    action = env.action_space.sample()
    obs, reward, done, truncated, info = env.step(action)

    env.render()

env.close()
