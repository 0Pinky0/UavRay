"""PyTorch policy class used for DQN"""

from typing import Dict, List, Tuple

import gymnasium as gym
from ray.rllib.algorithms.dqn.dqn_tf_policy import (
    PRIO_WEIGHTS,
    Q_SCOPE,
    Q_TARGET_SCOPE,
    postprocess_nstep_and_prio,
)
from ray.rllib.models.catalog import ModelCatalog
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.torch.torch_action_dist import (
    get_torch_categorical_class_with_temperature,
    TorchDistributionWrapper,
)
from ray.rllib.policy.policy import Policy
from ray.rllib.policy.policy_template import build_policy_class
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy.torch_mixins import (
    LearningRateSchedule,
    TargetNetworkMixin,
)

from ray.rllib.utils.error import UnsupportedSpaceException
from ray.rllib.utils.exploration.parameter_noise import ParameterNoise
from ray.rllib.utils.torch_utils import (
    apply_grad_clipping,
    concat_multi_gpu_td_errors,
    FLOAT_MIN,
    huber_loss,
    l2_loss,
    reduce_mean_ignore_inf,
)
from ray.rllib.utils.typing import TensorType, AlgorithmConfigDict

from model.d3qn_model import ComplexInputQNet
from online.dqn_model import D3QNTorchModel

import online.dqn

import torch
import torch.nn.functional as F


class QWithActionPredictionLoss:
    def __init__(
            self,
            q_t_selected: TensorType,
            q_tp1_best: TensorType,
            importance_weights: TensorType,
            rewards: TensorType,
            done_mask: TensorType,
            predicted_action: TensorType,
            one_hot_selection: TensorType,
            id_ratio: float = 0.2,
            gamma=0.99,
            n_step=1,
            loss_fn=huber_loss,
    ):
        q_tp1_best_masked = (1.0 - done_mask) * q_tp1_best

        # compute RHS of bellman equation
        q_t_selected_target = rewards + gamma ** n_step * q_tp1_best_masked

        inverse_dynamics_loss = F.cross_entropy(
            input=predicted_action,
            target=one_hot_selection,
        )

        # compute the error (potentially clipped)
        self.td_error = q_t_selected - q_t_selected_target.detach()
        self.loss = (torch.mean(importance_weights.float() * loss_fn(self.td_error))
                     + id_ratio * inverse_dynamics_loss)
        self.stats = {
            "mean_q": torch.mean(q_t_selected),
            "min_q": torch.min(q_t_selected),
            "max_q": torch.max(q_t_selected),
            "id_loss": inverse_dynamics_loss,
        }


class ComputeTDErrorMixin:
    """Assign the `compute_td_error` method to the DQNTorchPolicy

    This allows us to prioritize on the worker side.
    """

    def __init__(self):
        def compute_td_error(
                obs_t, act_t, rew_t, obs_tp1, terminateds_mask, importance_weights
        ):
            input_dict = self._lazy_tensor_dict({SampleBatch.CUR_OBS: obs_t})
            input_dict[SampleBatch.ACTIONS] = act_t
            input_dict[SampleBatch.REWARDS] = rew_t
            input_dict[SampleBatch.NEXT_OBS] = obs_tp1
            input_dict[SampleBatch.TERMINATEDS] = terminateds_mask
            input_dict[PRIO_WEIGHTS] = importance_weights

            # Do forward pass on loss to update td error attribute
            build_q_losses(self, self.model, None, input_dict)

            return self.model.tower_stats["q_loss"].td_error

        self.compute_td_error = compute_td_error


def build_q_model_and_distribution(
        policy: Policy,
        obs_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        config: AlgorithmConfigDict,
) -> Tuple[ModelV2, TorchDistributionWrapper]:
    """Build q_model and target_model for DQN

    Args:
        policy: The policy, which will use the model for optimization.
        obs_space (gym.spaces.Space): The policy's observation space.
        action_space (gym.spaces.Space): The policy's action space.
        config (AlgorithmConfigDict):

    Returns:
        (q_model, TorchCategorical)
            Note: The target q model will not be returned, just assigned to
            `policy.target_model`.
    """
    if not isinstance(action_space, gym.spaces.Discrete):
        raise UnsupportedSpaceException(
            "Action space {} is not supported for DQN.".format(action_space)
        )

    if config["hiddens"]:
        # try to infer the last layer size, otherwise fall back to 256
        num_outputs = ([256] + list(config["model"]["fcnet_hiddens"]))[-1]
        config["model"]["no_final_linear"] = True
    else:
        num_outputs = action_space.n

    model = ModelCatalog.get_model_v2(
        obs_space=obs_space,
        action_space=action_space,
        num_outputs=num_outputs,
        model_config=config["model"],
        name=Q_SCOPE,
        framework="torch",
        default_model=ComplexInputQNet,
        # q_hiddens=config["hiddens"],
        # dueling=config["dueling"],
    )

    policy.target_model = ModelCatalog.get_model_v2(
        obs_space=obs_space,
        action_space=action_space,
        num_outputs=num_outputs,
        model_config=config["model"],
        name=Q_TARGET_SCOPE,
        framework="torch",
        default_model=ComplexInputQNet,
        # q_hiddens=config["hiddens"],
        # dueling=config["dueling"],
    )

    # Return a Torch TorchCategorical distribution where the temperature
    # parameter is partially binded to the configured value.
    temperature = config["categorical_distribution_temperature"]

    return model, get_torch_categorical_class_with_temperature(temperature)


def get_distribution_inputs_and_class(
        policy: Policy,
        model: ModelV2,
        input_dict: SampleBatch,
        *,
        explore: bool = True,
        is_training: bool = False,
        **kwargs
) -> Tuple[TensorType, type, List[TensorType]]:
    q_vals = compute_q_values(
        policy, model, input_dict, explore=explore, is_training=is_training
    )
    q_vals = q_vals[0] if isinstance(q_vals, tuple) else q_vals

    model.tower_stats["q_values"] = q_vals

    # Return a Torch TorchCategorical distribution where the temperature
    # parameter is partially binded to the configured value.
    temperature = policy.config["categorical_distribution_temperature"]

    return (
        q_vals,
        get_torch_categorical_class_with_temperature(temperature),
        [],  # state-out
    )


def build_q_losses(policy: Policy, model: ComplexInputQNet, _, train_batch: SampleBatch) -> TensorType:
    """Constructs the loss for DQNTorchPolicy.

    Args:
        policy: The Policy to calculate the loss for.
        model (ModelV2): The Model to calculate the loss for.
        train_batch: The training data.

    Returns:
        TensorType: A single loss tensor.
    """

    config = policy.config
    # Q-network evaluation.
    embed_t = model.encoder(
        train_batch[SampleBatch.CUR_OBS]['observation'],
        train_batch[SampleBatch.CUR_OBS]['raster']
    )
    q_t = model.q_head(embed_t)

    # Target Q-network evaluation.
    embed_tp1_target = policy.target_models[model].encoder(
        train_batch[SampleBatch.NEXT_OBS]['observation'],
        train_batch[SampleBatch.NEXT_OBS]['raster']
    )
    q_tp1 = policy.target_models[model].q_head(embed_tp1_target)

    # Q scores for actions which we know were selected in the given state.
    one_hot_selection = F.one_hot(
        train_batch[SampleBatch.ACTIONS].long(), policy.action_space.n
    )
    q_t_selected = torch.sum(
        torch.where(q_t > FLOAT_MIN, q_t, torch.tensor(0.0, device=q_t.device))
        * one_hot_selection,
        1,
    )

    # compute estimate of best possible value starting from state at t + 1
    embed_tp1 = model.encoder(
        train_batch[SampleBatch.NEXT_OBS]['observation'],
        train_batch[SampleBatch.NEXT_OBS]['raster']
    )
    q_tp1_using_online_net = model.q_head(embed_tp1)
    q_tp1_best_using_online_net = torch.argmax(q_tp1_using_online_net, 1)
    q_tp1_best_one_hot_selection = F.one_hot(
        q_tp1_best_using_online_net, policy.action_space.n
    )
    q_tp1_best = torch.sum(
        torch.where(
            q_tp1 > FLOAT_MIN, q_tp1, torch.tensor(0.0, device=q_tp1.device)
        )
        * q_tp1_best_one_hot_selection,
        1,
    )

    loss_fn = huber_loss if policy.config["td_error_loss_fn"] == "huber" else l2_loss

    predicted_action = model.inverse_dynamics(
        torch.cat([embed_t, embed_tp1], dim=-1)
    )

    q_loss = QWithActionPredictionLoss(
        q_t_selected,
        q_tp1_best,
        train_batch[PRIO_WEIGHTS],
        train_batch[SampleBatch.REWARDS],
        train_batch[SampleBatch.TERMINATEDS].float(),
        predicted_action,
        one_hot_selection.float(),
        config["id_ratio"],
        config["gamma"],
        config["n_step"],
        loss_fn,
    )

    # Store values for stats function in model (tower), such that for
    # multi-GPU, we do not override them during the parallel loss phase.
    model.tower_stats["td_error"] = q_loss.td_error
    # TD-error tensor in final stats
    # will be concatenated and retrieved for each individual batch item.
    model.tower_stats["q_loss"] = q_loss

    return q_loss.loss


def adam_optimizer(
        policy: Policy, config: AlgorithmConfigDict
) -> "torch.optim.Optimizer":
    # By this time, the models have been moved to the GPU - if any - and we
    # can define our optimizers using the correct CUDA variables.
    if not hasattr(policy, "q_func_vars"):
        policy.q_func_vars = policy.model.variables()

    return torch.optim.Adam(
        policy.q_func_vars, lr=policy.cur_lr, eps=config["adam_epsilon"]
    )


def build_q_stats(policy: Policy, batch) -> Dict[str, TensorType]:
    stats = {}
    for stats_key in policy.model_gpu_towers[0].tower_stats["q_loss"].stats.keys():
        stats[stats_key] = torch.mean(
            torch.stack(
                [
                    t.tower_stats["q_loss"].stats[stats_key].to(policy.device)
                    for t in policy.model_gpu_towers
                    if "q_loss" in t.tower_stats
                ]
            )
        )
    stats["cur_lr"] = policy.cur_lr
    return stats


def setup_early_mixins(
        policy: Policy, obs_space, action_space, config: AlgorithmConfigDict
) -> None:
    LearningRateSchedule.__init__(policy, config["lr"], config["lr_schedule"])


def before_loss_init(
        policy: Policy,
        obs_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        config: AlgorithmConfigDict,
) -> None:
    ComputeTDErrorMixin.__init__(policy)
    TargetNetworkMixin.__init__(policy)


def compute_q_values(
        policy: Policy,
        model: D3QNTorchModel,
        input_dict,
        state_batches=None,
        seq_lens=None,
        explore=None,
        is_training: bool = False,
):
    config = policy.config

    model_out, state = model(input_dict, state_batches or [], seq_lens)

    (action_scores, logits, probs_or_logits) = model.get_q_value_distributions(
        model_out
    )

    state_score = model.get_state_value(model_out)

    advantages_mean = reduce_mean_ignore_inf(action_scores, 1)
    advantages_centered = action_scores - torch.unsqueeze(advantages_mean, 1)
    value = state_score + advantages_centered

    return value, logits, probs_or_logits, state, model_out


def grad_process_and_td_error_fn(
        policy: Policy, optimizer: "torch.optim.Optimizer", loss: TensorType
) -> Dict[str, TensorType]:
    # Clip grads if configured.
    return apply_grad_clipping(policy, optimizer, loss)


def extra_action_out_fn(
        policy: Policy, input_dict, state_batches, model, action_dist
) -> Dict[str, TensorType]:
    return {"q_values": model.tower_stats["q_values"]}


D3QNTorchPolicy = build_policy_class(
    name="DQNTorchPolicy",
    framework="torch",
    loss_fn=build_q_losses,
    get_default_config=lambda: online.dqn.D3QNConfig(),
    make_model_and_action_dist=build_q_model_and_distribution,
    action_distribution_fn=get_distribution_inputs_and_class,
    stats_fn=build_q_stats,
    postprocess_fn=postprocess_nstep_and_prio,
    optimizer_fn=adam_optimizer,
    extra_grad_process_fn=grad_process_and_td_error_fn,
    extra_learn_fetches_fn=concat_multi_gpu_td_errors,
    extra_action_out_fn=extra_action_out_fn,
    before_init=setup_early_mixins,
    before_loss_init=before_loss_init,
    mixins=[
        TargetNetworkMixin,
        ComputeTDErrorMixin,
        LearningRateSchedule,
    ],
)
