import gymnasium.envs.registration

gymnasium.envs.registration.register(
    id="UavEnv-v7",
    entry_point="uav_envs.uav_env_v7:UavEnvironment",
)
