import numpy as np
import gym
from gym import spaces
import cv2
from gym_super_mario_bros.actions import RIGHT_ONLY

# =========================================================
# FRAME SKIP
# =========================================================
class SkipFrame(gym.Wrapper):
    """
    Repeat action for N frames and accumulate reward.
    """
    def __init__(self, env, skip=4):
        super().__init__(env)
        self.skip = skip

    def step(self, action):
        total_reward = 0.0
        terminated = False
        truncated = False
        info = {}

        for _ in range(self.skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward

            if terminated or truncated:
                break

        return obs, total_reward, terminated, truncated, info


# =========================================================
# GRAYSCALE
# =========================================================
class GrayScaleObservation(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)

        h, w, _ = self.observation_space.shape

        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(h, w, 1),
            dtype=np.uint8
        )

    def observation(self, obs):
        gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        return np.expand_dims(gray, axis=-1)


# =========================================================
# RESIZE
# =========================================================
class ResizeObservation(gym.ObservationWrapper):
    def __init__(self, env, shape=84):
        super().__init__(env)

        self.shape = (shape, shape)

        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(shape, shape, 1),
            dtype=np.uint8
        )

    def observation(self, obs):
        resized = cv2.resize(obs, self.shape, interpolation=cv2.INTER_AREA)
        return np.expand_dims(resized.squeeze(), axis=-1)


# =========================================================
# FRAME STACK (FIXED)
# =========================================================
class FrameStack(gym.Wrapper):
    """
    Stack last N frames: output shape (N, H, W)
    """
    def __init__(self, env, num_stack=4):
        super().__init__(env)

        self.num_stack = num_stack

        h, w, c = env.observation_space.shape

        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(num_stack, h, w),
            dtype=np.uint8
        )

        self.frames = np.zeros((num_stack, h, w), dtype=np.uint8)

    def reset(self, **kwargs):
        result = self.env.reset(**kwargs)

        if isinstance(result, tuple):
            obs, info = result
        else:
            obs, info = result, {}

        frame = obs[:, :, 0]

        self.frames[:] = frame

        return self.frames.copy(), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        frame = obs[:, :, 0]

        self.frames[:-1] = self.frames[1:]
        self.frames[-1] = frame

        return self.frames.copy(), reward, terminated, truncated, info


# =========================================================
# ENV CREATION
# =========================================================
def make_mario_env(render=False):
    import warnings
    warnings.filterwarnings("ignore")

    import gym_super_mario_bros
    from gym_super_mario_bros.actions import RIGHT_ONLY
    from nes_py.wrappers import JoypadSpace
    from gym.wrappers import TimeLimit   # <-- ADD THIS

    base_env = gym_super_mario_bros.make(
        "SuperMarioBros-1-1-v3",
        apply_api_compatibility=True,
        render_mode="human" if render else None,
    )

    env = JoypadSpace(base_env, RIGHT_ONLY)

    env = SkipFrame(env, skip=4)
    env = GrayScaleObservation(env)
    env = ResizeObservation(env, shape=84)
    env = FrameStack(env, num_stack=4)
    env = StuckDetection(env, max_no_progress=200)
    # -------------------------------
    # 🔥 LIMIT EPISODE LENGTH HERE
    # -------------------------------
    env = TimeLimit(env, max_episode_steps=4000)

    return env

class StuckDetection(gym.Wrapper):
    def __init__(self, env, max_no_progress=200):
        super().__init__(env)
        self.max_no_progress = max_no_progress
        self.last_x = 0
        self.stuck_counter = 0

    def reset(self, **kwargs):
        self.last_x = 0
        self.stuck_counter = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        x = info.get("x_pos", None)

        if x is not None:
            if x <= self.last_x:
                self.stuck_counter += 1
            else:
                self.stuck_counter = 0
                self.last_x = x

        # FORCE RESET if stuck too long
        if self.stuck_counter > self.max_no_progress:
            truncated = True  # end episode early
            info["stuck_reset"] = True

        return obs, reward, terminated, truncated, info