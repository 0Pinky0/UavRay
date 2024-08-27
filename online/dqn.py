"""
Deep Q-Networks (DQN, Rainbow, Parametric DQN)
==============================================

This file defines the distributed Algorithm class for the Deep Q-Networks
algorithm. See `dqn_[tf|torch]_policy.py` for the definition of the policies.

Detailed documentation:
https://docs.ray.io/en/master/rllib-algorithms.html#deep-q-networks-dqn-rainbow-parametric-dqn
"""  # noqa: E501

import logging
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

import numpy as np
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.algorithms.algorithm_config import AlgorithmConfig, NotProvided
from ray.rllib.execution.rollout_ops import (
    synchronous_parallel_sample,
)
from ray.rllib.execution.train_ops import (
    train_one_step,
    multi_gpu_train_one_step,
)
from ray.rllib.policy.policy import Policy
from ray.rllib.policy.sample_batch import MultiAgentBatch
from ray.rllib.utils import deep_update
from ray.rllib.utils.annotations import override
from ray.rllib.utils.deprecation import DEPRECATED_VALUE
from ray.rllib.utils.metrics import (
    LAST_TARGET_UPDATE_TS,
    NUM_AGENT_STEPS_SAMPLED,
    NUM_ENV_STEPS_SAMPLED,
    NUM_TARGET_UPDATES,
    SAMPLE_TIMER,
    SYNCH_WORKER_WEIGHTS_TIMER,
)
from ray.rllib.utils.replay_buffers.utils import sample_min_n_steps_from_buffer
from ray.rllib.utils.replay_buffers.utils import (
    update_priorities_in_replay_buffer,
    validate_buffer_config,
)
from ray.rllib.utils.typing import ResultDict
from ray.rllib.utils.typing import SampleBatchType

logger = logging.getLogger(__name__)


class D3QNConfig(AlgorithmConfig):
    def __init__(self, algo_class=None):
        """Initializes a D3QNConfig instance."""
        super().__init__(algo_class=algo_class or D3QN)

        # Overrides of AlgorithmConfig defaults
        # `env_runners()`
        # Set to `self.n_step`, if 'auto'.
        self.rollout_fragment_length: Union[int, str] = "auto"
        self.exploration_config = {
            "type": "EpsilonGreedy",
            "initial_epsilon": 1.0,
            "final_epsilon": 0.02,
            "epsilon_timesteps": 10000,
        }
        # New stack uses `epsilon` as either a constant value or a scheduler
        # defined like this.
        # TODO (simon): Ensure that users can understand how to provide epsilon.
        #  (sven): Should we add this to `self.env_runners(epsilon=..)`?
        self.epsilon = [(0, 1.0), (10000, 0.05)]

        # `training()`
        self.grad_clip = 40.0
        # Note: Only when using enable_rl_module_and_learner=True can the clipping mode
        # be configured by the user. On the old API stack, RLlib will always clip by
        # global_norm, no matter the value of `grad_clip_by`.
        self.grad_clip_by = "global_norm"
        self.lr = 5e-4
        self.train_batch_size = 32

        # `evaluation()`
        self.evaluation(evaluation_config=AlgorithmConfig.overrides(explore=False))

        # `reporting()`
        self.min_time_s_per_iteration = None
        self.min_sample_timesteps_per_iteration = 1000

        # DQN specific config settings.
        # fmt: off
        # __sphinx_doc_begin__
        self.target_network_update_freq = 500
        self.num_steps_sampled_before_learning_starts = 1000
        self.store_buffer_in_checkpoints = False
        self.lr_schedule = None
        self.adam_epsilon = 1e-8

        self.tau = 1.0

        self.dueling = True
        self.hiddens = [256]
        self.double_q = True
        self.n_step = 1
        self.before_learn_on_batch = None
        self.training_intensity = None
        self.td_error_loss_fn = "huber"
        self.categorical_distribution_temperature = 1.0
        self.id_ratio = 0.25

        # Replay buffer configuration.
        self.replay_buffer_config = {
            "type": "MultiAgentPrioritizedReplayBuffer",
            # Specify prioritized replay by supplying a buffer type that supports
            # prioritization, for example: MultiAgentPrioritizedReplayBuffer.
            "prioritized_replay": DEPRECATED_VALUE,
            # Size of the replay buffer. Note that if async_updates is set,
            # then each worker will have a replay buffer of this size.
            "capacity": 50000,
            "prioritized_replay_alpha": 0.6,
            # Beta parameter for sampling from prioritized replay buffer.
            "prioritized_replay_beta": 0.4,
            # Epsilon to add to the TD errors when updating priorities.
            "prioritized_replay_eps": 1e-6,
            # The number of continuous environment steps to replay at once. This may
            # be set to greater than 1 to support recurrent models.
            "replay_sequence_length": 1,
            # Whether to compute priorities on workers.
            "worker_side_prioritization": False,
        }
        # fmt: on
        # __sphinx_doc_end__

        # Deprecated.
        self.buffer_size = DEPRECATED_VALUE
        self.prioritized_replay = DEPRECATED_VALUE
        self.learning_starts = DEPRECATED_VALUE
        self.replay_batch_size = DEPRECATED_VALUE
        # Can not use DEPRECATED_VALUE here because -1 is a common config value
        self.replay_sequence_length = None
        self.prioritized_replay_alpha = DEPRECATED_VALUE
        self.prioritized_replay_beta = DEPRECATED_VALUE
        self.prioritized_replay_eps = DEPRECATED_VALUE

    @override(AlgorithmConfig)
    def training(
            self,
            *,
            target_network_update_freq: Optional[int] = NotProvided,
            replay_buffer_config: Optional[dict] = NotProvided,
            store_buffer_in_checkpoints: Optional[bool] = NotProvided,
            lr_schedule: Optional[List[List[Union[int, float]]]] = NotProvided,
            adam_epsilon: Optional[float] = NotProvided,
            grad_clip: Optional[int] = NotProvided,
            num_steps_sampled_before_learning_starts: Optional[int] = NotProvided,
            tau: Optional[float] = NotProvided,
            dueling: Optional[bool] = NotProvided,
            hiddens: Optional[int] = NotProvided,
            double_q: Optional[bool] = NotProvided,
            n_step: Optional[Union[int, Tuple[int, int]]] = NotProvided,
            id_ratio: Optional[float] = NotProvided,
            before_learn_on_batch: Callable[
                [Type[MultiAgentBatch], List[Type[Policy]], Type[int]],
                Type[MultiAgentBatch],
            ] = NotProvided,
            training_intensity: Optional[float] = NotProvided,
            td_error_loss_fn: Optional[str] = NotProvided,
            categorical_distribution_temperature: Optional[float] = NotProvided,
            **kwargs,
    ) -> "D3QNConfig":
        # Pass kwargs onto super's `training()` method.
        super().training(**kwargs)

        if target_network_update_freq is not NotProvided:
            self.target_network_update_freq = target_network_update_freq
        if replay_buffer_config is not NotProvided:
            # Override entire `replay_buffer_config` if `type` key changes.
            # Update, if `type` key remains the same or is not specified.
            new_replay_buffer_config = deep_update(
                {"replay_buffer_config": self.replay_buffer_config},
                {"replay_buffer_config": replay_buffer_config},
                False,
                ["replay_buffer_config"],
                ["replay_buffer_config"],
            )
            self.replay_buffer_config = new_replay_buffer_config["replay_buffer_config"]
        if store_buffer_in_checkpoints is not NotProvided:
            self.store_buffer_in_checkpoints = store_buffer_in_checkpoints
        if lr_schedule is not NotProvided:
            self.lr_schedule = lr_schedule
        if adam_epsilon is not NotProvided:
            self.adam_epsilon = adam_epsilon
        if grad_clip is not NotProvided:
            self.grad_clip = grad_clip
        if num_steps_sampled_before_learning_starts is not NotProvided:
            self.num_steps_sampled_before_learning_starts = (
                num_steps_sampled_before_learning_starts
            )
        if tau is not NotProvided:
            self.tau = tau
        if dueling is not NotProvided:
            self.dueling = dueling
        if hiddens is not NotProvided:
            self.hiddens = hiddens
        if double_q is not NotProvided:
            self.double_q = double_q
        if n_step is not NotProvided:
            self.n_step = n_step
        if id_ratio is not NotProvided:
            self.id_ratio = id_ratio
        if before_learn_on_batch is not NotProvided:
            self.before_learn_on_batch = before_learn_on_batch
        if training_intensity is not NotProvided:
            self.training_intensity = training_intensity
        if td_error_loss_fn is not NotProvided:
            self.td_error_loss_fn = td_error_loss_fn
        if categorical_distribution_temperature is not NotProvided:
            self.categorical_distribution_temperature = (
                categorical_distribution_temperature
            )

        return self

    @override(AlgorithmConfig)
    def validate(self) -> None:
        # Call super's validation method.
        super().validate()

        if (
                not self.enable_rl_module_and_learner
                and self.exploration_config["type"] == "ParameterNoise"
        ):
            if self.batch_mode != "complete_episodes":
                raise ValueError(
                    "ParameterNoise Exploration requires `batch_mode` to be "
                    "'complete_episodes'. Try setting `config.env_runners("
                    "batch_mode='complete_episodes')`."
                )

        if not self.enable_env_runner_and_connector_v2 and not self.in_evaluation:
            validate_buffer_config(self)

        if self.td_error_loss_fn not in ["huber", "mse"]:
            raise ValueError("`td_error_loss_fn` must be 'huber' or 'mse'!")

        # Check rollout_fragment_length to be compatible with n_step.
        if (
                not self.in_evaluation
                and self.rollout_fragment_length != "auto"
                and self.rollout_fragment_length < self.n_step
        ):
            raise ValueError(
                f"Your `rollout_fragment_length` ({self.rollout_fragment_length}) is "
                f"smaller than `n_step` ({self.n_step})! "
                "Try setting config.env_runners(rollout_fragment_length="
                f"{self.n_step})."
            )

        # TODO (simon): Find a clean solution to deal with
        # configuration configs when using the new API stack.
        if (
                not self.enable_rl_module_and_learner
                and self.exploration_config["type"] == "ParameterNoise"
        ):
            if self.batch_mode != "complete_episodes":
                raise ValueError(
                    "ParameterNoise Exploration requires `batch_mode` to be "
                    "'complete_episodes'. Try setting `config.env_runners("
                    "batch_mode='complete_episodes')`."
                )

        # Validate that we use the corresponding `EpisodeReplayBuffer` when using
        # episodes.
        # TODO (sven, simon): Implement the multi-agent case for replay buffers.
        from ray.rllib.utils.replay_buffers.episode_replay_buffer import (
            EpisodeReplayBuffer,
        )

        if (
                self.enable_env_runner_and_connector_v2
                and not isinstance(self.replay_buffer_config["type"], str)
                and not issubclass(self.replay_buffer_config["type"], EpisodeReplayBuffer)
        ):
            raise ValueError(
                "When using the new `EnvRunner API` the replay buffer must be of type "
                "`EpisodeReplayBuffer`."
            )

    @override(AlgorithmConfig)
    def get_rollout_fragment_length(self, worker_index: int = 0) -> int:
        if self.rollout_fragment_length == "auto":
            return (
                self.n_step[1]
                if isinstance(self.n_step, (tuple, list))
                else self.n_step
            )
        else:
            return self.rollout_fragment_length

    @property
    @override(AlgorithmConfig)
    def _model_config_auto_includes(self) -> Dict[str, Any]:
        return super()._model_config_auto_includes | {
            "double_q": self.double_q,
            "dueling": self.dueling,
            "epsilon": self.epsilon,
        }


def calculate_rr_weights(config: AlgorithmConfig) -> List[float]:
    """Calculate the round robin weights for the rollout and train steps"""
    if not config.training_intensity:
        return [1, 1]

    # Calculate the "native ratio" as:
    # [train-batch-size] / [size of env-rolled-out sampled data]
    # This is to set freshly rollout-collected data in relation to
    # the data we pull from the replay buffer (which also contains old
    # samples).
    native_ratio = config.train_batch_size / (
            config.get_rollout_fragment_length()
            * config.num_envs_per_env_runner
            # Add one to workers because the local
            # worker usually collects experiences as well, and we avoid division by zero.
            * max(config.num_env_runners + 1, 1)
    )

    # Training intensity is specified in terms of
    # (steps_replayed / steps_sampled), so adjust for the native ratio.
    sample_and_train_weight = config.training_intensity / native_ratio
    if sample_and_train_weight < 1:
        return [int(np.round(1 / sample_and_train_weight)), 1]
    else:
        return [1, int(np.round(sample_and_train_weight))]


class D3QN(Algorithm):
    @classmethod
    @override(Algorithm)
    def get_default_config(cls) -> AlgorithmConfig:
        return D3QNConfig()

    @classmethod
    @override(Algorithm)
    def get_default_policy_class(
            cls, config: AlgorithmConfig
    ) -> Optional[Type[Policy]]:
        from online.dqn_policy import D3QNTorchPolicy
        return D3QNTorchPolicy

    @override(Algorithm)
    def training_step(self) -> ResultDict:
        """DQN training iteration function.

        Each training iteration, we:
        - Sample (MultiAgentBatch) from workers.
        - Store new samples in replay buffer.
        - Sample training batch (MultiAgentBatch) from replay buffer.
        - Learn on training batch.
        - Update remote workers' new policy weights.
        - Update target network every `target_network_update_freq` sample steps.
        - Return all collected metrics for the iteration.

        Returns:
            The results dict from executing the training iteration.
        """
        return self._training_step_old_and_hybrid_api_stack()

    def _training_step_old_and_hybrid_api_stack(self) -> ResultDict:
        """Training step for the old and hybrid training stacks.

        More specifically this training step relies on `RolloutWorker`.
        """
        train_results = {}

        # We alternate between storing new samples and sampling and training
        store_weight, sample_and_train_weight = calculate_rr_weights(self.config)

        for _ in range(store_weight):
            # Sample (MultiAgentBatch) from workers.
            with self._timers[SAMPLE_TIMER]:
                new_sample_batch: SampleBatchType = synchronous_parallel_sample(
                    worker_set=self.workers, concat=True
                )

            # Update counters
            self._counters[NUM_AGENT_STEPS_SAMPLED] += new_sample_batch.agent_steps()
            self._counters[NUM_ENV_STEPS_SAMPLED] += new_sample_batch.env_steps()

            # Store new samples in replay buffer.
            self.local_replay_buffer.add(new_sample_batch)

        global_vars = {
            "timestep": self._counters[NUM_ENV_STEPS_SAMPLED],
        }

        # Update target network every `target_network_update_freq` sample steps.
        cur_ts = self._counters[
            (
                NUM_AGENT_STEPS_SAMPLED
                if self.config.count_steps_by == "agent_steps"
                else NUM_ENV_STEPS_SAMPLED
            )
        ]

        if cur_ts > self.config.num_steps_sampled_before_learning_starts:
            for _ in range(sample_and_train_weight):
                # Sample training batch (MultiAgentBatch) from replay buffer.
                train_batch = sample_min_n_steps_from_buffer(
                    self.local_replay_buffer,
                    self.config.train_batch_size,
                    count_by_agent_steps=self.config.count_steps_by == "agent_steps",
                )

                # Postprocess batch before we learn on it
                post_fn = self.config.get("before_learn_on_batch") or (lambda b, *a: b)
                train_batch = post_fn(train_batch, self.workers, self.config)

                # Learn on training batch.
                # Use simple optimizer (only for multi-agent or tf-eager; all other
                # cases should use the multi-GPU optimizer, even if only using 1 GPU)
                if self.config.get("simple_optimizer") is True:
                    train_results = train_one_step(self, train_batch)
                else:
                    train_results = multi_gpu_train_one_step(self, train_batch)

                # Update replay buffer priorities.
                update_priorities_in_replay_buffer(
                    self.local_replay_buffer,
                    self.config,
                    train_batch,
                    train_results,
                )

                last_update = self._counters[LAST_TARGET_UPDATE_TS]
                if cur_ts - last_update >= self.config.target_network_update_freq:
                    to_update = self.workers.local_worker().get_policies_to_train()
                    self.workers.local_worker().foreach_policy_to_train(
                        lambda p, pid, to_update=to_update: (
                                pid in to_update and p.update_target()
                        )
                    )
                    self._counters[NUM_TARGET_UPDATES] += 1
                    self._counters[LAST_TARGET_UPDATE_TS] = cur_ts

                # Update weights and global_vars - after learning on the local worker -
                # on all remote workers.
                with self._timers[SYNCH_WORKER_WEIGHTS_TIMER]:
                    self.workers.sync_weights(global_vars=global_vars)

        # Return all collected metrics for the iteration.
        return train_results
