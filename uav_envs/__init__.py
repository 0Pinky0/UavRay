import gymnasium.envs.registration

gymnasium.envs.registration.register(
    id="UavEnv-v1",
    entry_point="uav_envs.uav_env_v1:UAVEnvironment",
)
gymnasium.envs.registration.register(
    id="UavEnv-v2",
    entry_point="uav_envs.uav_envs_v2:UAVEnvironment",
)
gymnasium.envs.registration.register(
    id="UavEnv-v3",
    entry_point="uav_envs.uav_env_v3:UavEnvironment",
)
gymnasium.envs.registration.register(
    id="UavEnv-v5",
    entry_point="uav_envs.uav_env_v5:UAVEnvironment",
)
gymnasium.envs.registration.register(
    id="UavEnv-v6",
    entry_point="uav_envs.uav_env_v6:UAVEnvironment",
)
gymnasium.envs.registration.register(
    id="UavEnv-v7",
    entry_point="uav_envs.uav_env_v7:UAVEnvironment",
)
