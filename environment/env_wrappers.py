import gym_super_mario_bros
import gym
from gym.spaces import Box
from PIL import Image
import numpy as np
import time


class FrameSkipEnv(gym.Wrapper):
    def __init__(self, env, frame_skip, viewer=None):
        super().__init__(env)
        self.env = env
        self.frame_skip = frame_skip
        self.viewer = viewer

    def step(self, action):
        reward = 0.0
        last_frame_time = 0
        target_fps = 1/60

        for i_frame_skip in range(self.frame_skip):
            state, r, done, info = self.env.step(action)
            reward += r

            if self.viewer:
                current_frame_time = time.time()
                while last_frame_time + target_fps > current_frame_time:
                    current_frame_time = time.time()
                last_frame_time = current_frame_time
                self.viewer.show(self.env.unwrapped.screen)

            if done:
                break

        return state, reward, done, info


class SmbRender(gym.ObservationWrapper):
    def __init__(self, env, shape=(84,84)):
        super().__init__(env)
        self.observation_space = Box(low=0, high=255, shape=shape, dtype=np.uint8)
    
    def observation(self, render):
        render = np.dot(render, [0.299, 0.587, 0.114])
        render = np.array(Image.fromarray(render[32:, :]).resize((84, 84)))
        render = render / np.linalg.norm(render)
        return render


class RewardWrapper(gym.RewardWrapper):
    def __init__(self, env, clip_reward=False):
        super().__init__(env)
        self.clip_reward = clip_reward
    
    def reward(self, r):
        if self.clip_reward:
            r = np.sign(r)
        return r