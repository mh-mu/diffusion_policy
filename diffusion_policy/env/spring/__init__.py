from gym.envs.registration import register

register(
    id='spring-keypoints-v0',
    entry_point='envs.spring.spring_keypoints_env:SprinKeypointsEnv',
    max_episode_steps=200,
    reward_threshold=1.0
)