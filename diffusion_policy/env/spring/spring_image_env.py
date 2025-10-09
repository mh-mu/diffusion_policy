import gym
from gym import spaces
import numpy as np
from diffusion_policy.env.spring.spring_env import SpringEnv


class SpringImageEnv(SpringEnv):
    def __init__(self,
                 render_size=96,
                 **kwargs):
        super().__init__(render_size=render_size, **kwargs)
        
        # Override observation space for image-based observations
        self.observation_space = spaces.Dict({
            'image': spaces.Box(
                low=0,
                high=1,
                shape=(3, render_size, render_size),
                dtype=np.float32
            ),
            'agent_pos': spaces.Box(
                low=0,
                high=512,
                shape=(2,),
                dtype=np.float32
            )
        })

    def _get_obs(self):
        # Get image observation
        img = self._render_frame(mode='rgb_array')
        # Normalize to [0, 1] and transpose to CHW format
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))  # HWC to CHW
        
        # Get agent position
        agent_pos = np.array(self.agent.position, dtype=np.float32)
        
        return {
            'image': img,
            'agent_pos': agent_pos
        }
