"""
Based on PureJaxRL Implementation of PPO
"""

import os
from pathlib import Path

import csv
import wandb.sdk

from gymnax_exchange.jaxob.config_io import load_config_from_file, save_config_to_file
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.95"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "true"
# os.environ["JAX_CHECK_TRACER_LEAKS"] = "true"
# os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
import logging

# Suppress Orbax logging
logging.getLogger('orbax').setLevel(logging.ERROR)
logging.getLogger('absl').setLevel(logging.ERROR)

import time
import jax # type: ignorepip 
jax.config.update('jax_disable_jit', False)

import jax.numpy as jnp # type: ignore
import flax.linen as nn
import numpy as np
import optax
from flax.linen.initializers import constant, orthogonal # type: ignore
from typing import Sequence, NamedTuple, Any, Dict, Optional, Callable
from flax.training.train_state import TrainState
from flax.training import orbax_utils
from flax.core import freeze, unfreeze
import distrax
import orbax.checkpoint as oxcp
import hydra
from omegaconf import DictConfig, OmegaConf
import gc
from dataclasses import replace,fields
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

#from jaxmarl.wrappers.baselines import SMAXLogWrapper
#from jaxmarl.environments.smax import map_name_to_scenario, HeuristicEnemySMAX
from gymnax_exchange.jaxen.marl_env import MARLEnv
from gymnax_exchange.jaxob.jaxob_config import MultiAgentConfig,Execution_EnvironmentConfig, World_EnvironmentConfig,MarketMaking_EnvironmentConfig,CONFIG_OBJECT_DICT

import wandb
import functools
import matplotlib.pyplot as plt

import sys
import datetime

class ScannedRNN(nn.Module):
    @functools.partial(
        nn.scan,
        variable_broadcast="params",
        in_axes=0,
        out_axes=0,
        split_rngs={"params": False},
    )
    @nn.compact
    def __call__(self, carry, x):
        """Applies the module."""
        rnn_state = carry
        ins, resets = x
        rnn_state = jnp.where(
            resets[:, jnp.newaxis],
            self.initialize_carry(*rnn_state.shape),
            rnn_state,
        )
        new_rnn_state, y = nn.GRUCell(features=ins.shape[1])(rnn_state, ins)
        return new_rnn_state, y

    @staticmethod
    def initialize_carry(batch_size, hidden_size):
        # Use a dummy key since the default state init fn is just zeros.
        cell = nn.GRUCell(features=hidden_size)
        return cell.initialize_carry(jax.random.PRNGKey(0), (batch_size, hidden_size))


def _resolve_activation(config: Dict) -> Callable[[jnp.ndarray], jnp.ndarray]:
    activation_cfg = config.get("ACTIVATION", "relu")
    if isinstance(activation_cfg, (list, tuple)) and len(activation_cfg) > 0:
        activation_cfg = activation_cfg[0]
    activation_name = str(activation_cfg).lower()
    if activation_name == "tanh":
        return nn.tanh
    if activation_name == "gelu":
        return nn.gelu
    return nn.relu


class TransformerEncoderBlock(nn.Module):
    embed_dim: int
    num_heads: int
    mlp_dim: int
    activation_fn: Callable[[jnp.ndarray], jnp.ndarray]

    @nn.compact
    def __call__(self, x):
        x_norm = nn.LayerNorm()(x)
        attn = nn.SelfAttention(
            num_heads=self.num_heads,
            qkv_features=self.embed_dim,
            out_features=self.embed_dim,
            dropout_rate=0.0,
        )(x_norm, deterministic=True)
        x = x + attn

        y = nn.LayerNorm()(x)
        y = nn.Dense(self.mlp_dim, kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0))(y)
        y = self.activation_fn(y)
        y = nn.Dense(self.embed_dim, kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0))(y)
        return x + y
    
class MultiActionOutputIndependant(nn.Module):
    action_dims: Sequence[int]
    config: Dict

    @nn.compact
    def __call__(self, x):
        # Create multiple output heads
        if isinstance(self.action_dims, (list, tuple)):
            # Multi-output case: create separate heads for each output
            action_logits_list = []
            for dim in self.action_dims:
                logits = nn.Dense(
                    dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
                )(x)
                action_logits_list.append(logits)

            pi = MultiCategorical(action_logits_list)
        else:
            raise ValueError("action_dims must be a list or tuple for MultiActionOutputIndependant.")

        return pi

class MultiActionOutputAutoregressive(nn.Module):
    action_dims: Sequence[int]
    config: Dict
    embed_dim: int = 32

    def get_logits_for_action(self, x, action_idx, prev_actions):
        """
        Compute logits for action_idx conditioned on prev_actions.
        
        Args:
            x: actor features (batch, feature_dim)
            action_idx: which action we're computing logits for (0, 1, 2, ...)
            prev_actions: list of previously sampled actions [action_0, action_1, ...]
        """
        if action_idx == 0:
            # First action: no conditioning
            logits = nn.Dense(
                self.action_dims[0], 
                kernel_init=orthogonal(0.01), 
                bias_init=constant(0.0),
                name=f'action_{action_idx}_head'
            )(x)
            return logits
        
        # Subsequent actions: condition on previous actions
        embeddings = []
        for i, prev_action in enumerate(prev_actions):
            # Embed each previous action
            embed = nn.Embed(
                num_embeddings=self.action_dims[i],
                features=self.embed_dim,
                name=f'action_{i}_embed'
            )(prev_action)
            embeddings.append(embed)
        
        # Concatenate features with all previous action embeddings
        combined = jnp.concatenate([x] + embeddings, axis=-1)
        
        # Process through hidden layer
        hidden = nn.Dense(
            self.config["GRU_HIDDEN_DIM"] // 2,
            kernel_init=orthogonal(2),
            bias_init=constant(0.0),
            name=f'action_{action_idx}_hidden'
        )(combined)
        hidden = nn.relu(hidden)
        
        # Output logits
        logits = nn.Dense(
            self.action_dims[action_idx],
            kernel_init=orthogonal(0.01),
            bias_init=constant(0.0),
            name=f'action_{action_idx}_head'
        )(hidden)
        
        return logits

    @nn.compact
    def __call__(self, x, given_actions=None):
        """
        Compute autoregressive action distribution.
        
        Args:
            x: actor features from the network
            given_actions: Optional. If provided (during training), use these for conditioning.
                          Shape: (..., num_actions). Otherwise sample autoregressively.
        
        Returns:
            AutoregressiveMultiCategorical distribution object
        """
        if not isinstance(self.action_dims, (list, tuple)):
            raise ValueError("action_dims must be a list or tuple for MultiActionOutputAutoregressive.")
        
        # Return a distribution that can sample autoregressively or compute log_prob
        return AutoregressiveMultiCategorical(
            actor_features=x,
            action_dims=self.action_dims,
            logits_fn=self.get_logits_for_action,
            given_actions=given_actions
        )

class SingleActionOutput(nn.Module):
    action_dim: int
    config: Dict

    @nn.compact
    def __call__(self, x):
        # Create multiple output heads
        if isinstance(self.action_dim, int):
            actor_mean = nn.Dense(
                self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0) # type: ignore
            )(x)
            # Avail actions are not used in the current implementation, but can be added if needed.
            # unavail_actions = 1 - avail_actions
            action_logits = actor_mean # - (unavail_actions * 1e10)
            pi = distrax.Categorical(logits=action_logits)
        else:
            raise ValueError("action_dims must be a list or tuple for MultiActionOutputIndependant.")

        return pi

class ActorCriticRNN(nn.Module):
    action_dim: Sequence[int]
    config: Dict
    actor_only: bool = False

    @nn.compact
    def __call__(self, hidden, x):
        # obs, dones, avail_actions = x
        obs, dones = x
        activation_fn = _resolve_activation(self.config)
        architecture = str(self.config.get("ARCHITECTURE", "rnn")).lower()

        base_fc_dim = int(self.config["FC_DIM_SIZE"])
        base_gru_dim = int(self.config["GRU_HIDDEN_DIM"])
        wide_factor = int(self.config.get("WIDE_FACTOR", 2))
        deep_layers = int(self.config.get("DEEP_LAYERS", 3))

        if architecture == "rnn_wide":
            pre_dim = base_fc_dim * wide_factor
            post_dim = base_gru_dim * wide_factor
        else:
            pre_dim = base_fc_dim
            post_dim = base_gru_dim

        embedding = obs
        if architecture == "rnn_deep":
            for layer_idx in range(max(1, deep_layers)):
                embedding = nn.Dense(
                    pre_dim,
                    kernel_init=orthogonal(jnp.sqrt(2)),
                    bias_init=constant(0.0),
                    name=f"obs_embedding_{layer_idx}",
                )(embedding)
                embedding = activation_fn(embedding)
            rnn_in = (embedding, dones)
            hidden, embedding = ScannedRNN()(hidden, rnn_in)
        elif architecture == "transformer":
            model_dim = int(self.config.get("TRANSFORMER_MODEL_DIM", pre_dim))
            num_layers = int(self.config.get("TRANSFORMER_NUM_LAYERS", 2))
            num_heads = int(self.config.get("TRANSFORMER_NUM_HEADS", 4))
            mlp_dim = int(self.config.get("TRANSFORMER_MLP_DIM", model_dim * 2))

            embedding = nn.Dense(
                model_dim,
                kernel_init=orthogonal(jnp.sqrt(2)),
                bias_init=constant(0.0),
                name="obs_embedding_transformer",
            )(embedding)
            embedding = activation_fn(embedding)

            # Transformer expects [batch, seq, feat]; inputs arrive as [seq, batch, feat]
            embedding_bt = jnp.swapaxes(embedding, 0, 1)
            dones_bt = jnp.swapaxes(dones, 0, 1)
            valid_mask = (1.0 - dones_bt)[..., jnp.newaxis]
            embedding_bt = embedding_bt * valid_mask

            for block_idx in range(num_layers):
                embedding_bt = TransformerEncoderBlock(
                    embed_dim=model_dim,
                    num_heads=num_heads,
                    mlp_dim=mlp_dim,
                    activation_fn=activation_fn,
                    name=f"transformer_block_{block_idx}",
                )(embedding_bt)

            embedding = jnp.swapaxes(embedding_bt, 0, 1)
            # Keep hidden for API compatibility with rollout code
            hidden = hidden
        else:
            embedding = nn.Dense(
                pre_dim,
                kernel_init=orthogonal(jnp.sqrt(2)),
                bias_init=constant(0.0),
                name="obs_embedding",
            )(embedding)
            embedding = activation_fn(embedding)
            rnn_in = (embedding, dones)
            hidden, embedding = ScannedRNN()(hidden, rnn_in)

        actor_mean = nn.Dense(
            post_dim,
            kernel_init=orthogonal(2),
            bias_init=constant(0.0),
            name="actor_hidden",
        )(embedding)

        actor_mean = activation_fn(actor_mean)

        # Normalize single-element lists to int (for n_actions=1 case)
        action_dim = self.action_dim
        if isinstance(action_dim, (list, tuple)) and len(action_dim) == 1:
            action_dim = action_dim[0]

        # Option 1: Single action output (current behavior)
        if isinstance(action_dim, int):
            pi = SingleActionOutput(action_dim=action_dim, config=self.config)(actor_mean)

        # Option 2: Multiple independent actions
        elif isinstance(action_dim, (list, tuple)):
            pi = MultiActionOutputIndependant(action_dims=action_dim, config=self.config)(actor_mean)

        # Option 3: Multiple autoregressive actions
        elif isinstance(action_dim, (list, tuple)) and self.config.get("AUTOREGRESSIVE", True):
            pi = MultiActionOutputAutoregressive(
                action_dims=action_dim,  # e.g., [10, 10, 5]
                config=self.config
            )(actor_mean)
        else:
            raise ValueError("action_dims must be int or list/tuple for ActorCriticRNN.")

        if self.actor_only:
            return hidden, pi

        critic = nn.Dense(
            post_dim,
            kernel_init=orthogonal(2),
            bias_init=constant(0.0),
            name="critic_hidden",
        )(embedding)
        critic = activation_fn(critic)
        critic = nn.Dense(
            1,
            kernel_init=orthogonal(1.0),
            bias_init=constant(0.0),
            name="critic_out",
        )(critic)

        return hidden, pi, jnp.squeeze(critic, axis=-1)


class MultiCategorical():
    """Wrapper for multiple independent categorical distributions.
    NOTE: The correct thing would be to let it inherit from distrax.Distribution but
    this requires additional thought to implement all abstract methods, many of which are not 
    needed for this use case. """
    
    def __init__(self, logits_list):
        self.categoricals = [distrax.Categorical(logits=logits) for logits in logits_list]

    
    def sample(self, seed):
        keys = jax.random.split(seed, len(self.categoricals))
        samples = [cat.sample(seed=key) for cat, key in zip(self.categoricals, keys)]
        return jnp.stack(samples, axis=-1)  # Shape: (..., num_outputs)
    
    def log_prob(self, actions):
        # actions should have shape (..., num_outputs)
        log_probs = [cat.log_prob(actions[...,i]) for i, cat in enumerate(self.categoricals)]
        return jnp.sum(jnp.stack(log_probs, axis=-1), axis=-1)  # Sum log probs for independence
    
    def mode(self):
        modes = [cat.mode() for cat in self.categoricals]
        return jnp.stack(modes, axis=-1)

    def entropy(self):
        entropies = [cat.entropy() for cat in self.categoricals]
        return jnp.sum(jnp.stack(entropies, axis=-1), axis=-1)  # Sum entropies for independence


class AutoregressiveMultiCategorical():
    """
    Wrapper for multiple categorical distributions where later actions 
    are conditioned on previously sampled actions.
    
    During sampling: samples actions sequentially, feeding each into the next.
    During training: computes conditional log probabilities using given actions.
    """
    
    def __init__(self, actor_features, action_dims, logits_fn, given_actions=None):
        """
        Args:
            actor_features: base features from the network (batch, feature_dim)
            action_dims: list of action space sizes, e.g., [10, 10, 5]
            logits_fn: function(x, action_idx, prev_actions) -> logits
            given_actions: optional actions to condition on (for training)
                          Shape: (..., num_actions)
        """
        self.actor_features = actor_features
        self.action_dims = action_dims
        self.logits_fn = logits_fn
        self.given_actions = given_actions
    
    def sample(self, seed):
        """Sample actions autoregressively."""
        keys = jax.random.split(seed, len(self.action_dims))
        samples = []
        
        for i, key in enumerate(keys):
            # Get logits conditioned on previously sampled actions
            logits = self.logits_fn(self.actor_features, i, samples)
            action = distrax.Categorical(logits=logits).sample(seed=key)
            samples.append(action)
        
        return jnp.stack(samples, axis=-1)  # Shape: (..., num_actions)
    
    def log_prob(self, actions):
        """
        Compute log probability of action sequence.
        Uses chain rule: log p(a1,a2,a3) = log p(a1) + log p(a2|a1) + log p(a3|a1,a2)
        
        Args:
            actions: action sequence, shape (..., num_actions)
        """
        log_probs = []
        
        for i in range(len(self.action_dims)):
            # Get previous actions for conditioning
            prev_actions = [actions[..., j] for j in range(i)]
            
            # Get conditional logits
            logits = self.logits_fn(self.actor_features, i, prev_actions)
            
            # Compute log prob of this action given previous ones
            log_p = distrax.Categorical(logits=logits).log_prob(actions[..., i])
            log_probs.append(log_p)
        
        # Sum log probabilities (chain rule)
        return jnp.sum(jnp.stack(log_probs, axis=-1), axis=-1)

    def mode(self):
        """Get the most likely sequence."""
        modes = []
        for i in range(len(self.action_dims)):
            logits = self.logits_fn(self.actor_features, i, modes)
            modes.append(jnp.argmax(logits, axis=-1))
        return jnp.stack(modes, axis=-1)

    def entropy(self):
        """
        Compute entropy of the autoregressive distribution.
        For autoregressive models: H = sum of conditional entropies
        """
        entropies = []
        
        # For entropy, we need to marginalize over previous actions
        # Simplified: compute entropy of each conditional separately
        # (This is an approximation - true entropy requires marginalization)
        for i in range(len(self.action_dims)):
            if self.given_actions is not None and i > 0:
                # Use given actions for conditioning
                prev_actions = [self.given_actions[..., j] for j in range(i)]
            else:
                # For first action or when no given actions, use empty list
                prev_actions = []
            
            logits = self.logits_fn(self.actor_features, i, prev_actions)
            entropy = distrax.Categorical(logits=logits).entropy()
            entropies.append(entropy)
        
        return jnp.sum(jnp.stack(entropies, axis=-1), axis=-1)

class Transition(NamedTuple):
    global_done: jnp.ndarray
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray
    # avail_actions: jnp.ndarray


def batchify(x: jnp.ndarray, num_actors):
    return x.reshape((num_actors, -1))


def unbatchify(x: jnp.ndarray,num_envs, num_agents):
    return  x.reshape((num_envs, num_agents, -1))

def create_agent_configs(config):
    """
    Create agent configs with three layers of precedence (lowest to highest):
    1. Default attributes from the EnvironmentConfig classes
    2. Values from the JSON config
    3. Sweep parameters from AGENT_CONFIGS

    Args:
        config: The full config dict containing both JSON config and sweep parameters
        config_dict: Dict mapping agent type names to their config classes
                    e.g., {"MarketMaking": MarketMaking_EnvironmentConfig, ...}

    Returns:
        Dict of agent configs keyed by agent type name
    """
    agent_configs = {}

    if "AGENT_CONFIGS" in config:
        for agent_type, agent_cfg in config["AGENT_CONFIGS"].items():
            # Start with defaults from the config class
            agent_config_class = CONFIG_OBJECT_DICT[agent_type]

            # First apply config values (from JSON) to override defaults
            config_overrides = {}
            field_names = {f.name for f in fields(agent_config_class)}
            for key, value in config["dict_of_agents_configs"].items():
                if  isinstance(value, dict) and key == agent_type:
                    for key, value in value.items():
                        if key in field_names:
                            config_overrides[key] = value

            # Then apply sweep parameters which take highest precedence
            sweep_overrides = {k: v for k, v in agent_cfg.items()}

            # Merge: sweep_overrides will override config_overrides
            all_overrides = {**config_overrides, **sweep_overrides}

            # Create the agent config with all overrides
            agent_configs[agent_type] = agent_config_class(**all_overrides)
    else:
        # No sweep overrides — use agent configs from the env config directly
        for agent_type, agent_cfg_dict in config.get("dict_of_agents_configs", {}).items():
            agent_config_class = CONFIG_OBJECT_DICT[agent_type]
            field_names = {f.name for f in fields(agent_config_class)}
            overrides = {k: v for k, v in agent_cfg_dict.items() if k in field_names}
            agent_configs[agent_type] = agent_config_class(**overrides)

    return agent_configs


def _get_npz_array(dataset: Any, primary_key: str, fallback_key: Optional[str] = None):
    if primary_key in dataset:
        return dataset[primary_key]
    if fallback_key is not None and fallback_key in dataset:
        return dataset[fallback_key]
    raise KeyError(f"Missing key in BC dataset: '{primary_key}'")


def _resolve_warmstart_checkpoint_dir(config: Dict) -> str:
    if config.get("WARMSTART_CHECKPOINT_DIR"):
        return str(config["WARMSTART_CHECKPOINT_DIR"])

    warm_project = config.get("WARMSTART_PROJECT")
    warm_run = config.get("WARMSTART_RUN")
    if warm_project and warm_run:
        return f'{config["world_config"]["alphatradePath"]}/checkpoints/MARLCheckpoints/{warm_project}/{warm_run}'

    raise ValueError(
        "Warm start requires WARMSTART_CHECKPOINT_DIR or both WARMSTART_PROJECT and WARMSTART_RUN."
    )


def _merge_params_by_key(target_params, source_params):
    target_dict = unfreeze(target_params)
    source_dict = unfreeze(source_params)

    def _merge(target_node, source_node):
        if isinstance(target_node, dict) and isinstance(source_node, dict):
            for key, value in source_node.items():
                if key in target_node:
                    target_node[key] = _merge(target_node[key], value)
            return target_node
        return source_node

    return freeze(_merge(target_dict, source_dict))


def make_train(config):
    # scenario = map_name_to_scenario(config["MAP_NAME"])
    init_key = jax.random.PRNGKey(config["SEED"])
    # Create a MultiAgentConfig object with parameters from the config
    print ("Overriding the Agent config objects with the variable from the sweep parameters.")
    agent_configs = create_agent_configs(config)
    
    ma_config = MultiAgentConfig(
        number_of_agents_per_type=config["NUM_AGENTS_PER_TYPE"],
        dict_of_agents_configs=agent_configs,
        world_config=World_EnvironmentConfig(
            seed=config["SEED"],
            timePeriod=str(config["TimePeriod"]),
            # Only override parameters that exist in both config and World_EnvironmentConfig
            **{k: v for k, v in config["world_config"].items()
            if hasattr(World_EnvironmentConfig(), k) and k not in ["seed",
                                                                    "timePeriod",
                                                                ]}
        ))
    print("The training environment config, after copying of sweep parameters is \n\t ",
          ma_config)


    # For evaluation, create a separate config with evaluation-specific parameters
    eval_ma_config = None
    if config["CALC_EVAL"]:
        # Reuse agent_configs from above if it exists
        eval_agent_configs = create_agent_configs(config)
        eval_ma_config = MultiAgentConfig(
                    number_of_agents_per_type=config["NUM_AGENTS_PER_TYPE"],
                    dict_of_agents_configs=eval_agent_configs,
                    world_config=World_EnvironmentConfig(
                        seed=config["SEED"],
                        timePeriod=str(config["EvalTimePeriod"]),
                        # Only override parameters that exist in both config and World_EnvironmentConfig
                        **{k: v for k, v in config["world_config"].items() 
                        if hasattr(World_EnvironmentConfig(), k) and k not in ["seed",
                                                                               "timePeriod",
                                                                               ]}
                    ))
   

    env : MARLEnv = MARLEnv(key=init_key, multi_agent_config=ma_config)
    if config["CALC_EVAL"]:
        eval_env: MARLEnv = MARLEnv(key=init_key,multi_agent_config=eval_ma_config)
    else:
        eval_env = None

    agent_type_names = list(env.type_names)

    config["NUM_ACTORS_PERTYPE"] = [n * config["NUM_ENVS"] for n in config["NUM_AGENTS_PER_TYPE"]]  # Should be a list.
    config["NUM_ACTORS_TOTAL"] = env.num_agents * config["NUM_ENVS"]

    config["NUM_UPDATES"] = int(
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )
    config["MINIBATCH_SIZES"] = [
        nact * config["NUM_STEPS"] // config["NUM_MINIBATCHES"] for i,nact in enumerate(config["NUM_ACTORS_PERTYPE"])
    ]
    # config["CLIP_EPS"] = (
    #     config["CLIP_EPS"] / env.num_agents
    #     if config["SCALE_CLIP_EPS"]
    #     else config["CLIP_EPS"]
    # )

    # env = SMAXLogWrapper(env)

    def linear_schedule(lr,count):
        frac = (
            1.0
            - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"]))
            / config["NUM_UPDATES"]
        )
        return lr * frac

    def train(rng, run: wandb.sdk.wandb_run.Run = None):
        training_mode = str(config.get("TRAINING_MODE", "rl_cold")).lower()
        if training_mode not in {"bc", "rl_cold", "rl_warm"}:
            raise ValueError("TRAINING_MODE must be one of: bc, rl_cold, rl_warm")

        def _init_train_states(rng_in, actor_only: bool = False):
            hstates_local = []
            train_states_local = []
            init_dones_agents_local = []
            for i, _ in enumerate(env.instance_list):
                network = ActorCriticRNN(env.action_spaces[i].n, config=config, actor_only=actor_only)
                rng_in, _rng_local = jax.random.split(rng_in)
                init_x_local = (
                    jnp.zeros((1, config["NUM_ENVS"], env.observation_spaces[i].shape[0])),
                    jnp.zeros((1, config["NUM_ENVS"])),
                )

                init_hstate_local = ScannedRNN.initialize_carry(config["NUM_ENVS"], config["GRU_HIDDEN_DIM"])
                network_params_local = network.init(_rng_local, init_hstate_local, init_x_local)
                if config["ANNEAL_LR"][i]:
                    tx = optax.chain(
                        optax.clip_by_global_norm(config["MAX_GRAD_NORM"][i]),
                        optax.adam(learning_rate=functools.partial(linear_schedule, config["LR"][i]), eps=1e-5),
                    )
                else:
                    tx = optax.chain(
                        optax.clip_by_global_norm(config["MAX_GRAD_NORM"][i]),
                        optax.adam(config["LR"][i], eps=1e-5),
                    )
                train_state_local = TrainState.create(
                    apply_fn=network.apply,
                    params=network_params_local,
                    tx=tx,
                )
                init_hstate_local = ScannedRNN.initialize_carry(config["NUM_ACTORS_PERTYPE"][i], config["GRU_HIDDEN_DIM"])
                hstates_local.append(init_hstate_local)
                train_states_local.append(train_state_local)
                init_dones_agents_local.append(jnp.zeros((config["NUM_ACTORS_PERTYPE"][i]), dtype=bool))
            return rng_in, hstates_local, train_states_local, init_dones_agents_local

        rng, hstates, train_states, init_dones_agents = _init_train_states(
            rng,
            actor_only=(training_mode == "bc"),
        )

        if training_mode == "bc":
            # Initialize environment parameters
            env_params = env.default_params
            
            # Behavior Cloning: Collect trajectories from environment using expert policies
            expert_policy = config.get("EXPERT_POLICY", "twap")
            bc_episodes = int(config.get("BC_EPISODES", 50))
            bc_epochs = int(config.get("BC_EPOCHS", 10))
            bc_batch_size = int(config.get("BC_BATCH_SIZE", 256))
            bc_log_every = int(config.get("BC_LOG_EVERY", 1))
            
            print(f"Behavior Cloning: Collecting {bc_episodes} episodes using {expert_policy.upper()} policy...")
            print(f"  WANDB_MODE={config['WANDB_MODE']}, run={run}, run is not None={run is not None}")
            
            # Collect trajectories from environment rollouts
            obs_buffer = {i: [] for i in range(len(env.instance_list))}
            action_buffer = {i: [] for i in range(len(env.instance_list))}
            done_buffer = {i: [] for i in range(len(env.instance_list))}
            episode_prices_slippage = {i: [] for i in range(len(env.instance_list))}
            per_step_slippage = {i: {step: [] for step in range(config["NUM_STEPS"])} for i in range(len(env.instance_list))}
            
            for episode_num in range(bc_episodes):
                rng, reset_key = jax.random.split(rng)
                reset_rng = jax.random.split(reset_key, config["NUM_ENVS"])
                obsv, env_state = jax.vmap(env.reset, in_axes=(0, None))(reset_rng, env_params)
                
                # Initialize hidden states and dones for all agents
                hstates_local = []
                init_dones_local = []
                for i in range(len(env.instance_list)):
                    hstates_local.append(ScannedRNN.initialize_carry(config["NUM_ENVS"], config["GRU_HIDDEN_DIM"]))
                    init_dones_local.append(jnp.zeros((config["NUM_ENVS"],), dtype=bool))
                
                for step_idx in range(config["NUM_STEPS"]):
                    rng, policy_key, step_key = jax.random.split(rng, 3)
                    step_keys = jax.random.split(step_key, config["NUM_ENVS"])
                    
                    # Get expert actions (using env's action space)
                    actions = []
                    for i, space in enumerate(env.action_spaces):
                        if expert_policy.lower() == "twap":
                            # TWAP: execute every 10 steps
                            twap_action = 1 if (step_idx % 10) == 0 else 0
                            action = jnp.full((config["NUM_ENVS"],), twap_action, dtype=jnp.int32)
                        elif expert_policy.lower() == "vwap":
                            # VWAP: front-loaded (70% in first half, 30% in second half)
                            if step_idx < 0.7 * config["NUM_STEPS"]:
                                vwap_action = 1
                            else:
                                vwap_action = 0
                            action = jnp.full((config["NUM_ENVS"],), vwap_action, dtype=jnp.int32)
                        else:
                            # Random fallback
                            subkeys = jax.random.split(policy_key, config["NUM_ENVS"])
                            action = jax.vmap(space.sample)(subkeys)
                        actions.append(action)
                    
                    # Store observations and actions
                    for i in range(len(env.instance_list)):
                        obs_buffer[i].append(np.asarray(obsv[i]))
                        action_buffer[i].append(np.asarray(actions[i]))
                    
                    # Step environment
                    obsv, env_state, reward, done, info = jax.vmap(
                        env.step, in_axes=(0, 0, 0, None)
                    )(step_keys, env_state, actions, env_params)
                    
                    # Store dones and collect price slippage at every step
                    for i in range(len(env.instance_list)):
                        done_np = np.asarray(done["agents"][i])
                        done_buffer[i].append(done_np)
                        
                        # Collect price slippage from every step (average later)
                        if "adj_slippage_bps" in info["agents"][i]:
                            adj_slip_np = np.asarray(info["agents"][i]["adj_slippage_bps"])
                            # adj_slip_np has shape (NUM_ENVS,), extract mean for this step
                            step_mean_slippage = float(np.mean(adj_slip_np))
                            episode_prices_slippage[i].append(step_mean_slippage)
                            # Also collect per-step slippage for statistics
                            if step_idx < config["NUM_STEPS"]:
                                per_step_slippage[i][step_idx].append(step_mean_slippage)
                    
                    # Reset on episode done
                    done_all = np.asarray(done["__all__"])
                    if np.all(done_all):
                        break
            
            print(f"BC Dataset Statistics:")
            
            # Print collected price slippage statistics
            for i in range(len(env.instance_list)):
                if len(episode_prices_slippage[i]) > 0:
                    mean_slippage = float(np.mean(episode_prices_slippage[i]))
                    std_slippage = float(np.std(episode_prices_slippage[i]))
                    print(f"  Agent {i}: collected {len(episode_prices_slippage[i])} steps with avg price_slippage={mean_slippage:.2f}bps (±{std_slippage:.2f}bps)")
                    # Print per-step slippage breakdown
                    print(f"    Per-step slippage for Agent {i}:")
                    for step in range(config["NUM_STEPS"]):
                        if per_step_slippage[i][step]:
                            step_slips = per_step_slippage[i][step]
                            step_mean = float(np.mean(step_slips))
                            step_std = float(np.std(step_slips))
                            step_min = float(np.min(step_slips))
                            step_max = float(np.max(step_slips))
                            print(f"      Step {step:3d}: mean={step_mean:7.2f}bps, std={step_std:6.2f}bps, min={step_min:7.2f}bps, max={step_max:7.2f}bps (n={len(step_slips)})")
                else:
                    print(f"  Agent {i}: NO price slippage data collected")
            
            # Create per-step slippage visualization
            fig, axes = plt.subplots(1, len(env.instance_list), figsize=(12 * len(env.instance_list), 6), squeeze=False)
            axes = axes[0]
            
            for i in range(len(env.instance_list)):
                ax = axes[i]
                if episode_prices_slippage[i]:
                    # Plot mean slippage across all collected steps
                    all_slippage = episode_prices_slippage[i]
                    steps = np.arange(len(all_slippage))
                    
                    # Plot clean line of mean slippage
                    ax.plot(steps, all_slippage, 'b-', linewidth=1.5, alpha=0.8)
                    
                    # Add zero line
                    ax.axhline(y=0, color='k', linestyle='--', alpha=0.3, linewidth=1)
                    
                    ax.set_xlabel('Collected Steps', fontsize=12)
                    ax.set_ylabel('Mean Slippage (bps)', fontsize=12)
                    ax.set_title(f'Agent {i}: Slippage Across All Collected Steps', fontsize=13, fontweight='bold')
                    ax.grid(True, alpha=0.3)
                
            plt.tight_layout()
            
            # Save figure
            fig_path = f'{config["world_config"]["alphatradePath"]}/bc_slippage_plot.png'
            plt.savefig(fig_path, dpi=150, bbox_inches='tight')
            print(f"✓ Saved per-step slippage plot to {fig_path}")
            
            # Log figure to wandb
            if config["WANDB_MODE"] != "disabled" and run is not None:
                try:
                    wandb.log({"bc/per_step_slippage_plot": wandb.Image(fig_path)}, commit=False)
                    print(f"✓ Logged per-step slippage plot to wandb")
                except Exception as e:
                    print(f"⚠ Failed to log plot to wandb: {e}")
            
            plt.close(fig)
            
            bc_losses = []
            
            # Global step counter across all agents and epochs
            global_step_for_bc = 0
            
            for i, train_state in enumerate(train_states):
                # Convert buffers to arrays
                obs = np.concatenate(obs_buffer[i], axis=0)  # (total_steps, num_envs, obs_dim)
                actions = np.concatenate(action_buffer[i], axis=0)  # (total_steps, num_envs)
                dones = np.concatenate(done_buffer[i], axis=0)  # (total_steps, num_envs)
                
                print(f"  Agent {i}: obs shape {obs.shape}, actions {actions.shape}, dones {dones.shape}")
                
                # Reshape for BC training
                obs = obs.reshape(-1, obs.shape[-1])  # Flatten environments and steps
                actions = actions.reshape(-1)
                dones = dones.reshape(-1)
                
                num_samples = len(actions)
                if num_samples == 0:
                    raise ValueError(f"BC dataset for agent {i} is empty")
                
                # Split train/val
                val_split = int(0.2 * num_samples)
                perm = np.random.permutation(num_samples)
                val_idx = perm[:val_split]
                train_idx = perm[val_split:]
                
                obs_train = obs[train_idx]
                actions_train = actions[train_idx]
                dones_train = dones[train_idx]
                
                obs_val = obs[val_idx]
                actions_val = actions[val_idx]
                dones_val = dones[val_idx]
                
                num_train_samples = len(train_idx)
                
                # Debug: Print class distribution
                unique_train, counts_train = np.unique(actions_train, return_counts=True)
                unique_val, counts_val = np.unique(actions_val, return_counts=True)
                print(f"  Agent {i} class distribution - Train: {dict(zip(unique_train, counts_train))}, Val: {dict(zip(unique_val, counts_val))}")
                class_dist_train = {int(k): int(v) for k, v in zip(unique_train, counts_train)}
                class_dist_val = {int(k): int(v) for k, v in zip(unique_val, counts_val)}
                
                # Compute class weights (inverse frequency) to handle imbalance
                n_class_0 = np.sum(actions_train == 0)
                n_class_1 = np.sum(actions_train == 1)
                weight_class_0 = n_class_1 / (n_class_0 + n_class_1) if n_class_0 > 0 else 1.0
                weight_class_1 = n_class_0 / (n_class_0 + n_class_1) if n_class_1 > 0 else 1.0
                print(f"  Agent {i} class weights - class_0: {weight_class_0:.4f}, class_1: {weight_class_1:.4f}")
                
                @jax.jit
                def _bc_update(train_state_in, obs_batch, done_batch, action_batch):
                    def _loss_fn(params):
                        # Reshape for time dimension
                        init_h = ScannedRNN.initialize_carry(obs_batch.shape[0], config["GRU_HIDDEN_DIM"])
                        obs_t = jnp.expand_dims(obs_batch, 0)  # Add time dim
                        done_t = jnp.expand_dims(done_batch, 0)
                        action_t = jnp.expand_dims(action_batch, 0)
                        _, pi = train_state_in.apply_fn(params, init_h, (obs_t, done_t))
                        log_probs = pi.log_prob(action_t)
                        
                        # Apply class weights: give higher weight to minority class
                        # action_batch is in range [0, num_actions-1]
                        weights = jnp.where(action_batch == 0, weight_class_0, weight_class_1)
                        weighted_log_probs = log_probs.squeeze(0) * weights
                        return -weighted_log_probs.mean()
                    
                    loss, grads = jax.value_and_grad(_loss_fn)(train_state_in.params)
                    train_state_out = train_state_in.apply_gradients(grads=grads)
                    return train_state_out, loss
                
                def _compute_bc_metrics(train_state_in, obs_batch, done_batch, action_batch):
                    """Compute accuracy, precision, recall, F1, and confusion matrix elements"""
                    init_h = ScannedRNN.initialize_carry(obs_batch.shape[0], config["GRU_HIDDEN_DIM"])
                    obs_t = jnp.expand_dims(obs_batch, 0)
                    done_t = jnp.expand_dims(done_batch, 0)
                    _, pi = train_state_in.apply_fn(train_state_in.params, init_h, (obs_t, done_t))
                    pred_actions = pi.mode().squeeze(0)
                    
                    # Accuracy
                    accuracy = float(jnp.mean(pred_actions == action_batch))
                    
                    # For binary classification metrics
                    pred_actions_np = np.asarray(pred_actions)
                    action_batch_np = np.asarray(action_batch)
                    
                    # Compute TP, FP, TN, FN
                    tp = float(np.sum((pred_actions_np == 1) & (action_batch_np == 1)))
                    fp = float(np.sum((pred_actions_np == 1) & (action_batch_np == 0)))
                    tn = float(np.sum((pred_actions_np == 0) & (action_batch_np == 0)))
                    fn = float(np.sum((pred_actions_np == 0) & (action_batch_np == 1)))
                    
                    # Precision
                    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                    
                    # Recall
                    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                    
                    # F1 Score
                    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
                    
                    # Specificity
                    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
                    
                    return {
                        "accuracy": accuracy,
                        "precision": precision,
                        "recall": recall,
                        "f1": f1,
                        "specificity": specificity,
                        "tp": tp,
                        "fp": fp,
                        "tn": tn,
                        "fn": fn,
                    }

            # Setup pure JAX evaluation function for actual model slippage
            def _bc_eval_step(eval_runner_state, unused):
                train_states_eval, eval_env_state, last_obs, last_done, h_states, eval_rng = eval_runner_state
                actions = []
                new_h_states = []
                for i_ag, train_state_eval in enumerate(train_states_eval):
                    obs_i = batchify(last_obs[i_ag], config["NUM_ACTORS_PERTYPE"][i_ag])
                    ac_in = (obs_i[jnp.newaxis, :], last_done[i_ag][jnp.newaxis, :])
                    h, pi = train_state_eval.apply_fn(train_state_eval.params, h_states[i_ag], ac_in)
                    action = pi.mode().squeeze(0)
                    action_unbatched = unbatchify(action, config["NUM_ENVS"], env.multi_agent_config.number_of_agents_per_type[i_ag])
                    actions.append(action_unbatched.squeeze())
                    new_h_states.append(h)
                    
                eval_rng, _rng = jax.random.split(eval_rng)
                rng_step = jax.random.split(_rng, config["NUM_ENVS"])
                
                obsv, eval_env_state, reward, done, info = jax.vmap(
                    env.step, in_axes=(0, 0, 0, None)
                )(rng_step, eval_env_state, actions, env_params)
                
                done_batch = []
                for i_ag in range(len(train_states_eval)):
                    d = batchify(done["agents"][i_ag], config["NUM_ACTORS_PERTYPE"][i_ag])
                    done_batch.append(jnp.squeeze(d))
                
                step_info = []
                for i_ag in range(len(train_states_eval)):
                    step_info.append(info["agents"][i_ag].get("adj_slippage_bps", jnp.zeros_like(reward[i_ag])))
                
                new_runner_state = (train_states_eval, eval_env_state, obsv, done_batch, new_h_states, eval_rng)
                return new_runner_state, step_info

            @jax.jit
            def _run_model_eval(train_states_current, eval_rng):
                eval_rng, reset_rng_key = jax.random.split(eval_rng)
                reset_rng = jax.random.split(reset_rng_key, config["NUM_ENVS"])
                eval_obsv, eval_env_state = jax.vmap(env.reset, in_axes=(0, None))(reset_rng, env_params)
                
                eval_hstates = []
                init_dones_agents_eval = []
                for i_ag in range(len(train_states_current)):
                    eval_hstates.append(ScannedRNN.initialize_carry(config["NUM_ACTORS_PERTYPE"][i_ag], config["GRU_HIDDEN_DIM"]))
                    init_dones_agents_eval.append(jnp.zeros((config["NUM_ACTORS_PERTYPE"][i_ag]), dtype=bool))
                    
                eval_runner_state_init = (train_states_current, eval_env_state, eval_obsv, init_dones_agents_eval, eval_hstates, eval_rng)
                _, eval_slippage_traj = jax.lax.scan(_bc_eval_step, eval_runner_state_init, None, config["NUM_STEPS"])
                return eval_slippage_traj

                
                for epoch in range(bc_epochs):
                    perm = np.random.permutation(num_train_samples)
                    losses_this_epoch = []
                    step_in_epoch = 0
                    
                    for start in range(0, num_train_samples, bc_batch_size):
                        batch_idx = perm[start:start + bc_batch_size]
                        obs_batch = jnp.asarray(obs_train[batch_idx])
                        done_batch = jnp.asarray(dones_train[batch_idx])
                        action_batch = jnp.asarray(actions_train[batch_idx])
                        train_state, loss = _bc_update(train_state, obs_batch, done_batch, action_batch)
                        loss_float = float(loss)
                        losses_this_epoch.append(loss_float)
                        
                        # Log every step/batch to wandb (just increment step, don't log metrics yet)
                        if config["WANDB_MODE"] != "disabled" and run is not None:
                            try:
                                # Just log epoch metadata without metrics
                                wandb.log({
                                    f"bc/agent_{agent_type_names[i]}/epoch": epoch + 1,
                                    f"bc/agent_{agent_type_names[i]}/step_in_epoch": step_in_epoch + 1,
                                    "bc/global_step": global_step_for_bc,
                                }, step=global_step_for_bc, commit=False)
                                global_step_for_bc += 1
                            except Exception as e:
                                print(f"  Warning: Failed to log batch to wandb: {e}")
                                global_step_for_bc += 1
                        else:
                            global_step_for_bc += 1
                        
                        step_in_epoch += 1
                    
                    epoch_loss = float(np.mean(losses_this_epoch))

                    # Evaluate ACTUAL model execution slippage
                    rng, _run_eval_rng = jax.random.split(rng)
                    eval_slippage_traj = _run_model_eval(train_states, _run_eval_rng)
                    
                    true_model_slippage_bps = None
                    if i < len(eval_slippage_traj):
                        true_model_slippage_bps = float(np.mean(np.asarray(eval_slippage_traj[i])))
                    
                    # Compute validation metrics every epoch
                    val_metrics = _compute_bc_metrics(train_state, jnp.asarray(obs_val), jnp.asarray(dones_val), jnp.asarray(actions_val))
                    
                    # Compute price slippage metric from validation episodes
                    price_slippage_bps = None
                    per_step_slippage_stats = {}
                    if len(episode_prices_slippage[i]) > 0:
                        # Average price slippage across all episodes for this agent
                        price_slippage_bps = float(np.mean(episode_prices_slippage[i]))
                        
                        # Compute per-step slippage statistics for logging
                        for step in range(config["NUM_STEPS"]):
                            if per_step_slippage[i][step]:
                                step_slips = per_step_slippage[i][step]
                                per_step_slippage_stats[step] = {
                                    "mean": float(np.mean(step_slips)),
                                    "std": float(np.std(step_slips)),
                                    "min": float(np.min(step_slips)),
                                    "max": float(np.max(step_slips)),
                                    "count": len(step_slips),
                                }
                    
                    # Log epoch summary to wandb
                    if config["WANDB_MODE"] != "disabled" and run is not None:
                        try:
                            log_dict = {
                                f"bc/agent_{agent_type_names[i]}/epoch_loss": epoch_loss,
                                f"bc/agent_{agent_type_names[i]}/val_accuracy": val_metrics["accuracy"],
                                f"bc/agent_{agent_type_names[i]}/val_precision": val_metrics["precision"],
                                f"bc/agent_{agent_type_names[i]}/val_recall": val_metrics["recall"],
                                f"bc/agent_{agent_type_names[i]}/val_f1": val_metrics["f1"],
                                f"bc/agent_{agent_type_names[i]}/val_specificity": val_metrics["specificity"],
                                f"bc/agent_{agent_type_names[i]}/progress": f"{epoch + 1}/{bc_epochs}",
                                "bc/current_epoch": epoch + 1,
                                "bc/total_bc_epochs": bc_epochs,
                            }
                            # Add price slippage metric if available
                            if price_slippage_bps is not None:
                            if true_model_slippage_bps is not None:
                                log_dict[f"bc/agent_{agent_type_names[i]}/price_slippage_bps"] = true_model_slippage_bps
                            wandb.log(log_dict, step=global_step_for_bc, commit=True)
                            global_step_for_bc += 1
                        except Exception as e:
                            print(f"  Warning: Failed to log epoch to wandb: {e}")
                            global_step_for_bc += 1
                    else:
                        global_step_for_bc += 1
                    
                    # Console output every bc_log_every for readability
                    if (epoch + 1) % bc_log_every == 0:
                        slippage_str = ""
                        if price_slippage_bps is not None:
                        if true_model_slippage_bps is not None:
                            slippage_str = f" true_model_slippage={true_model_slippage_bps:.2f}bps"
                            # Add per-step slippage summary
                            if per_step_slippage_stats:
                                step_means = [stats["mean"] for stats in per_step_slippage_stats.values()]
                                step_mean_avg = float(np.mean(step_means)) if step_means else 0.0
                                step_mean_min = float(np.min(step_means)) if step_means else 0.0
                                step_mean_max = float(np.max(step_means)) if step_means else 0.0
                                slippage_str += f" [per-step: avg={step_mean_avg:.2f}bps, min={step_mean_min:.2f}bps, max={step_mean_max:.2f}bps]"
                        print(f"BC agent {i} epoch {epoch + 1}/{bc_epochs} loss={epoch_loss:.6f} val_acc={val_metrics['accuracy']:.4f} val_f1={val_metrics['f1']:.4f}{slippage_str}")
                
                bc_losses.append(epoch_loss)
                train_states[i] = train_state
            
            # Roll out the trained BC model in the environment to calculate actual slippage
            print("Evaluating trained BC model in the environment to calculate true slippage...")
            
            # Use deterministic mode for evaluation
            def _bc_eval_step(eval_runner_state, unused):
                train_states_eval, eval_env_state, last_obs, last_done, h_states, eval_rng = eval_runner_state
                
                actions = []
                new_h_states = []
                
                for i_ag, train_state_eval in enumerate(train_states_eval):
                    obs_i = last_obs[i_ag]
                    # Expand dimensions for network input
                    obs_i = batchify(obs_i, config["NUM_ACTORS_PERTYPE"][i_ag])
                    ac_in = (
                        obs_i[jnp.newaxis, :],
                        last_done[i_ag][jnp.newaxis, :],
                    )
                    
                    # actor_only=True returns h_state, pi (no value)
                    h, pi = train_state_eval.apply_fn(train_state_eval.params, h_states[i_ag], ac_in)
                    action = pi.mode().squeeze(0)  # Use mode for deterministic eval
                    
                    action_unbatched = unbatchify(action, config["NUM_ENVS"], env.multi_agent_config.number_of_agents_per_type[i_ag])
                    actions.append(action_unbatched.reshape((config["NUM_ENVS"],)))
                    new_h_states.append(h)
                    
                eval_rng, _rng = jax.random.split(eval_rng)
                rng_step = jax.random.split(_rng, config["NUM_ENVS"])
                
                obsv, eval_env_state, reward, done, info = jax.vmap(
                    env.step, in_axes=(0, 0, 0, None)
                )(rng_step, eval_env_state, actions, env_params)
                
                done_batch = []
                for i_ag in range(len(train_states_eval)):
                    d = batchify(done["agents"][i_ag], config["NUM_ACTORS_PERTYPE"][i_ag])
                    done_batch.append(jnp.squeeze(d))
                
                # Extract slippage
                step_info = []
                for i_ag in range(len(train_states_eval)):
                    step_info.append(info["agents"][i_ag].get("adj_slippage_bps", jnp.zeros_like(reward[i_ag])))

                new_runner_state = (train_states_eval, eval_env_state, obsv, done_batch, new_h_states, eval_rng)
                return new_runner_state, step_info
            
            # Setup initial state for evaluation rollout
            rng, _rng = jax.random.split(rng)
            reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
            eval_obsv, eval_env_state = jax.vmap(env.reset, in_axes=(0, None))(reset_rng, env_params)
            
            eval_hstates = []
            init_dones_agents_eval = []
            for i_ag in range(len(train_states)):
                eval_hstates.append(ScannedRNN.initialize_carry(config["NUM_ACTORS_PERTYPE"][i_ag], config["GRU_HIDDEN_DIM"]))
                init_dones_agents_eval.append(jnp.zeros((config["NUM_ACTORS_PERTYPE"][i_ag]), dtype=bool))
                
            eval_runner_state = (
                train_states,
                eval_env_state,
                eval_obsv,
                init_dones_agents_eval,
                eval_hstates,
                _rng,
            )
            
            # Run scan 
            print("Jitting evaluation loop...")
            bc_eval_scan = jax.jit(lambda state: jax.lax.scan(_bc_eval_step, state, None, config["NUM_STEPS"]))
            _, eval_slippage_traj = bc_eval_scan(eval_runner_state)
            
            # Parse and log metrics
            for i_ag in range(len(train_states)):
                agent_slip = np.asarray(eval_slippage_traj[i_ag])
                actual_slippage = float(np.mean(agent_slip))
                print(f"Agent {i_ag} True Model Mean Slippage: {actual_slippage:.2f} bps")
                if config["WANDB_MODE"] != "disabled" and run is not None:
                    wandb.log({
                        f"bc/agent_{agent_type_names[i_ag]}/true_model_price_slippage_bps": actual_slippage,
                        f"bc/agent_{agent_type_names[i_ag]}/true_model_price_slippage_std": float(np.std(agent_slip))
                    }, commit=False)

            # Save checkpoint
            checkpoint_dir = f'{config["world_config"]["alphatradePath"]}/checkpoints/MARLCheckpoints/{config["PROJECT"]}/{(run.name if run and run.name else run.id) if run else "GENERIC_RUN"}'
            orbax_checkpointer = oxcp.PyTreeCheckpointer()
            checkpoint_manager = oxcp.CheckpointManager(
                checkpoint_dir,
                orbax_checkpointer,
                oxcp.CheckpointManagerOptions(max_to_keep=2, create=True),
            )
            bc_ckpt = {
                "model": train_states,
                "metrics": {"bc_losses": bc_losses},
            }
            save_args = orbax_utils.save_args_from_target(bc_ckpt)
            checkpoint_manager.save(bc_epochs, bc_ckpt, save_kwargs={"save_args": save_args})
            checkpoint_manager.wait_until_finished()
            
            # Log final BC summary metrics to wandb
            if config["WANDB_MODE"] != "disabled":
                if run is not None:
                    try:
                        final_metrics = {
                            "bc/total_epochs": bc_epochs,
                            "bc/total_episodes": bc_episodes,
                            "bc/final_loss": float(bc_losses[-1]) if bc_losses else 0.0,
                            "bc/mean_loss": float(np.mean(bc_losses)) if bc_losses else 0.0,
                        }
                        wandb.log(final_metrics, commit=True)
                        print(f"✓ Logged final BC metrics to wandb: {final_metrics}")
                    except Exception as e:
                        print(f"✗ Failed to log final metrics to wandb: {e}")
                else:
                    print(f"⚠ wandb run is None despite WANDB_MODE={config['WANDB_MODE']}")
            else:
                print(f"⚠ WANDB_MODE={config['WANDB_MODE']} (disabled)")
            
            print(f"BC training complete. Saved checkpoint to {checkpoint_dir}")
            return {"runner_state": (train_states,), "bc_losses": bc_losses}

        if training_mode == "rl_warm":
            warm_ckpt_dir = _resolve_warmstart_checkpoint_dir(config)
            orbax_checkpointer = oxcp.PyTreeCheckpointer()
            checkpoint_manager = oxcp.CheckpointManager(warm_ckpt_dir, orbax_checkpointer)
            warm_step = config.get("WARMSTART_CHECKPOINT_STEP", None)
            if warm_step is None:
                warm_step = checkpoint_manager.latest_step()
            if warm_step is None:
                raise ValueError(f"No checkpoint found in warm-start directory: {warm_ckpt_dir}")

            restored_state = checkpoint_manager.restore(int(warm_step))
            restored_train_states = restored_state["model"] if isinstance(restored_state, dict) and "model" in restored_state else restored_state
            train_states = [
                train_states[i].replace(
                    params=_merge_params_by_key(train_states[i].params, restored_train_states[i].params)
                )
                for i in range(len(train_states))
            ]
            print(f"Loaded warm-start params from {warm_ckpt_dir} at step {warm_step}")

        # INIT ENV
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
        env_params=env.default_params
        if config["CALC_EVAL"]:
            eval_env_params=eval_env.default_params # type: ignore
        else:
            eval_env_params = None
        # Multi-device setup
        devices = jax.devices()
        n_devices = len(devices)
        mesh = Mesh(np.array(devices), axis_names=('batch',))
        batch_sharding = NamedSharding(mesh, P('batch'))
        replicated_sharding = NamedSharding(mesh, P())
        assert config["NUM_ENVS"] % n_devices == 0, \
            f"NUM_ENVS ({config['NUM_ENVS']}) must be divisible by number of devices ({n_devices})"
        print(f"Multi-device: {n_devices} device(s) detected, sharding {config['NUM_ENVS']} envs across them")

        obsv, env_state = jax.vmap(env.reset, in_axes=(0,None))(reset_rng,env_params)
        # TRAIN LOOP
        

        def _update_step(update_runner_state,env_params,eval_env_params, unused):
            # COLLECT TRAJECTORIES
            runner_state, update_steps = update_runner_state
            def _env_step(runner_state, unused):
                train_states, env_state, last_obs, last_done,h_states, rng = runner_state

                # SELECT ACTION
                rng, _rng = jax.random.split(rng)
                
                # Ignore getting the available actions for now, assume all actions are available.
                # avail_actions = jax.vmap(env.get_avail_actions)(env_state.env_state)
                # avail_actions = jax.lax.stop_gradient(
                #     batchify(avail_actions, env.agents, config["NUM_ACTORS"])
                # )
                # obs_batch = batchify(last_obs, env.agents, config["NUM_ACTORS"])
                actions=[]
                values=[]
                log_probs=[]

                for i, train_state in enumerate(train_states):
                    obs_i= last_obs[i]
                    obs_i=batchify(obs_i,config["NUM_ACTORS_PERTYPE"][i])  # Reshape to match the input shape of the network
                    ac_in = (
                        obs_i[jnp.newaxis, :],
                        last_done[i][jnp.newaxis, :],
                        # avail_actions,
                    )
                    h_states[i], pi, value = train_state.apply_fn(train_state.params, h_states[i], ac_in)
                    values.append(value)
                    action = pi.sample(seed=_rng)
                    log_probs.append(pi.log_prob(action))
                    action=unbatchify(action, config["NUM_ENVS"], env.multi_agent_config.number_of_agents_per_type[i])  # Reshape to match the action shape
                    actions.append(action.squeeze())
                    # env_act = unbatchify(
                    #     action, env.agents, config["NUM_ENVS"], env.num_agents
                    # )
                    # env_act = {k: v.squeeze() for k, v in env_act.items()}
                # STEP ENV
                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config["NUM_ENVS"])

                obsv, env_state, reward, done, info = jax.vmap(
                    env.step, in_axes=(0, 0, 0,None)
                )(rng_step, env_state, actions,env_params)
                def reward_callback(reward_max_idx,reward,trades,bids,asks,p_vwap):
                    if reward[reward_max_idx] > 5:
                        print("Reward exceeded threshold:", reward[reward_max_idx])
                        # print("Observation:", obsv[reward_max_idx])
                        print("Trades ", trades[reward_max_idx])
                        print("Bids ", bids[reward_max_idx])
                        print("Asks ", asks[reward_max_idx])
                        print("P_vwap ", p_vwap[reward_max_idx])
                        # print("Info:", info_ag[reward_max_idx], info_world[reward_max_idx])
                    return reward
                # jax.debug.callback(reward_callback, jnp.argmax(jnp.abs(reward[0])), reward[0], env_state.world_state.trades, env_state.world_state.bid_raw_orders, env_state.world_state.ask_raw_orders,env_state.agent_states[0].p_vwap)

                # info = jax.tree.map(lambda x: x.reshape((config["NUM_ACTORS"])), info)
                
                done_batch=done
                transitions=[]
                for i,train_state in enumerate(train_states):
                    done_batch['agents'][i] = batchify(done["agents"][i],config["NUM_ACTORS_PERTYPE"][i]).squeeze()
                    obs_batch = batchify(last_obs[i],config["NUM_ACTORS_PERTYPE"][i])
                    action_batch = batchify(actions[i],config["NUM_ACTORS_PERTYPE"][i])
                    value = values[i]
                    log_prob = log_probs[i]

                    info_i={"world":info["world"],"agent":jax.tree.map(lambda x: x.reshape(config["NUM_ACTORS_PERTYPE"][i],-1),info["agents"][i])}
                    # print(f"info for agenttype {i}:", info_i)


                    transitions.append(Transition(
                        jnp.tile(done["__all__"], config["NUM_AGENTS_PER_TYPE"][i]),
                        last_done[i],
                        action_batch.squeeze(),
                        value.squeeze(),
                        batchify(reward[i], config["NUM_ACTORS_PERTYPE"][i]).squeeze(),
                        log_prob.squeeze(),
                        obs_batch,
                        info_i,
                        # avail_actions,
                    ))
                runner_state = (train_states, env_state, obsv, done_batch['agents'], h_states, rng)
                return runner_state, transitions

            initial_hstates = runner_state[-2]
            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, None, config["NUM_STEPS"]
            )

            # CALCULATE ADVANTAGE
            train_states, env_state, last_obs, last_dones, hstates_new, rng = runner_state

            def _calculate_gae(gamma,gae_lambda,traj_batch, last_val):
                def _get_advantages(gae_and_next_value, transition):
                    gae, next_value = gae_and_next_value
                    done, value, reward = (
                        transition.global_done,
                        transition.value,
                        transition.reward,
                    )
                    delta = reward + gamma * next_value * (1 - done) - value
                    gae = (
                        delta
                        + gamma * gae_lambda * (1 - done) * gae
                    )
                    return (gae, value), gae

                _, advantages = jax.lax.scan(
                    _get_advantages,
                    (jnp.zeros_like(last_val), last_val),
                    traj_batch,
                    reverse=True,
                    unroll=16,
                )
                return advantages, advantages + traj_batch.value

            advantages=[]
            targets=[]
            for i, train_state in enumerate(train_states):
                last_obs_batch = batchify(last_obs[i], config["NUM_ACTORS_PERTYPE"][i])
                # avail_actions = jnp.ones(
                #     (config["NUM_ACTORS"], env.action_space(env.agents[0]).n)
                # )
                ac_in = (
                    last_obs_batch[jnp.newaxis, :],
                    last_dones[i][jnp.newaxis, :],
                    # avail_actions,
                )
                _, _, last_val = train_state.apply_fn(train_state.params, hstates_new[i], ac_in)
                last_val = last_val.squeeze()

                advantages_i, targets_i = _calculate_gae(config["GAMMA"][i],config["GAE_LAMBDA"][i],traj_batch[i], last_val)
                advantages.append(advantages_i)
                targets.append(targets_i)

            # UPDATE NETWORKS
            loss_infos = []
            for i, train_state in enumerate(train_states):
                def _update_epoch(update_state, unused):
                    def _update_minbatch(train_state, batch_info):
                        init_hstate, traj_batch, advantages, targets = batch_info

                        def _loss_fn(params, init_hstate, traj_batch, gae, targets):
                            # RERUN NETWORK
                            _, pi, value = train_state.apply_fn(
                                params,
                                init_hstate.squeeze(),
                                (traj_batch.obs, traj_batch.done),
                            )
                            log_prob = pi.log_prob(traj_batch.action)
                            # CALCULATE VALUE LOSS
                            value_pred_clipped = traj_batch.value + (
                                value - traj_batch.value
                            ).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
                            value_losses = jnp.square(value - targets)
                            value_losses_clipped = jnp.square(value_pred_clipped - targets)
                            value_loss = 0.5 * jnp.maximum(
                                value_losses, value_losses_clipped
                            ).mean()

                            # CALCULATE ACTOR LOSS
                            logratio = log_prob - traj_batch.log_prob
                            # jax.debug.print("Log ratio for epoch max,min: {}", (jnp.max(logratio), jnp.min(logratio)))
                            # jax.debug.print("Log prob replay : {}, {}", log_prob.shape, log_prob[:3,:3])
                            # jax.debug.print("Log prob traj : {}, {}", traj_batch.log_prob.shape,traj_batch.log_prob[:3,:3])
                            ratio = jnp.exp(logratio)
                            gae = (gae - gae.mean()) / (gae.std() + 1e-8)
                            loss_actor1 = ratio * gae
                            loss_actor2 = (
                                jnp.clip(
                                    ratio,
                                    1.0 - config["CLIP_EPS"],
                                    1.0 + config["CLIP_EPS"],
                                )
                                * gae
                            )
                            loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
                            loss_actor = loss_actor.mean()
                            entropy = pi.entropy().mean()

                            # debug
                            approx_kl = ((ratio - 1) - logratio).mean()
                            clip_frac = jnp.mean(jnp.abs(ratio - 1) > config["CLIP_EPS"])

                            total_loss = (
                                loss_actor
                                + config["VF_COEF"][i] * value_loss
                                - config["ENT_COEF"][i] * entropy
                            )
                            return total_loss, (value_loss, loss_actor, entropy, ratio, approx_kl, clip_frac)

                        grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                        total_loss, grads = grad_fn(
                            train_state.params, init_hstate, traj_batch, advantages, targets
                        )
                        train_state = train_state.apply_gradients(grads=grads)
                        return train_state, total_loss
                    
                    (
                        train_state,
                        init_hstate,
                        traj_batch,
                        advantages,
                        targets,
                        rng,
                    ) = update_state
                    rng, _rng = jax.random.split(rng)

                    # adding an additional "fake" dimensionality to perform minibatching correctly
                    init_hstate = jnp.reshape(
                        init_hstate, (1, config["NUM_ACTORS_PERTYPE"][i], -1)
                    )
                    batch = (
                        init_hstate,
                        traj_batch,
                        advantages.squeeze(),
                        targets.squeeze(),
                    )
                    permutation = jax.random.permutation(_rng, config["NUM_ACTORS_PERTYPE"][i])
                    shuffled_batch = jax.tree.map(
                        lambda x: jnp.take(x, permutation, axis=1), batch
                    )

                    minibatches = jax.tree.map(
                        lambda x: jnp.swapaxes(
                            jnp.reshape(
                                x,
                                [x.shape[0], config["NUM_MINIBATCHES"], -1]
                                + list(x.shape[2:]),
                            ),
                            1,
                            0,
                        ),
                        shuffled_batch,
                    )

                    train_state, total_loss = jax.lax.scan(
                        _update_minbatch, train_state, minibatches
                    )
                    update_state = (
                        train_state,
                        init_hstate.squeeze(),
                        traj_batch,
                        advantages,
                        targets,
                        rng,
                    )
                    return update_state, total_loss

                update_state = (
                    train_state,
                    initial_hstates[i],
                    traj_batch[i],
                    advantages[i],
                    targets[i],
                    rng,
                )
                update_state, loss_info = jax.lax.scan(
                    _update_epoch, update_state, None, config["UPDATE_EPOCHS"]
                )
                train_states[i] = update_state[0]
                loss_infos.append(loss_info)


            metrics= {}
            metrics['agents'] = [jax.tree.map(
                lambda x: x.reshape(
                    (config["NUM_STEPS"], config["NUM_ENVS"], config["NUM_AGENTS_PER_TYPE"][i])
                ),
                trjbtch.info['agent']) for i, trjbtch in enumerate(traj_batch)]
            metrics['world'] = [traj_batch.info['world'] for i, traj_batch in enumerate(traj_batch)]
            metrics["loss"]=[]
            for i,loss_info in enumerate(loss_infos):
                # jax.debug.print("Ratio dimensions for 1st agent: {}", loss_info[1][3].shape)
                #Ratio dimensions are (epochs, num_minibatches, n_steps, num_envs)
                ratio_0 = loss_info[1][3].at[0,0].get().mean()
                ratio_other_epochs=loss_info[1][3].at[1:,:].get().mean()
                ratio_other_batches=loss_info[1][3].at[0,1:].get().mean()
                # jax.debug.print("Ratio first epoch, first minibatch: {}", ratio_0)
                # jax.debug.print("Ratio other epochs mean: {}", ratio_other_epochs)
                # jax.debug.print("Ratio other batches mean: {}", ratio_other_batches)
                loss_info = jax.tree.map(lambda x: x.mean(), loss_info)
                metrics["loss"].append({
                    "total_loss": loss_info[0],
                    "value_loss": loss_info[1][0],
                    "actor_loss": loss_info[1][1],
                    "entropy": loss_info[1][2],
                    "ratio": loss_info[1][3],
                    "ratio_0": ratio_0,
                    "ratio_other_epochs": ratio_other_epochs,
                    "ratio_other_batches": ratio_other_batches,
                    "approx_kl": loss_info[1][4],
                    "clip_frac": loss_info[1][5],
                    "weighted_entropy_loss": loss_info[1][2] * config["ENT_COEF"][i],
                    "weighted_value_loss": loss_info[1][0] * config["VF_COEF"][i],
                })
            metrics['avg_reward'] = [jnp.mean(tr.reward) for tr in traj_batch]
            metrics["traj_batch"] = traj_batch


            if config["CALC_EVAL"]:
                def _eval_step(eval_runner_state, unused):
                    train_states, eval_env_state, last_obs, last_done,h_states, rng = eval_runner_state
                    rng, _rng = jax.random.split(rng)
                
                    actions=[]
                    values=[]
                    log_probs=[]

                    for i, train_state in enumerate(train_states):
                        obs_i= last_obs[i]
                        obs_i=batchify(obs_i,config["NUM_ACTORS_PERTYPE"][i])  # Reshape to match the input shape of the network
                        ac_in = (
                            obs_i[jnp.newaxis, :],
                            last_done[i][jnp.newaxis, :],
                            # avail_actions,
                        )
                        h_states[i], pi, value = train_state.apply_fn(train_state.params, h_states[i], ac_in)
                        values.append(value)
                        action = pi.sample(seed=_rng)
                        log_probs.append(pi.log_prob(action))
                        action=unbatchify(action, config["NUM_ENVS"], env.multi_agent_config.number_of_agents_per_type[i])  # Reshape to match the action shape
                        actions.append(action.squeeze())

                        rng, _rng = jax.random.split(rng)
                        rng_step = jax.random.split(_rng, config["NUM_ENVS"])

                





                    # STEP ENV
                    rng, _rng = jax.random.split(rng)
                    rng_step = jax.random.split(_rng, config["NUM_ENVS"])
                    obsv, eval_env_state, reward, done, info = jax.vmap(
                        eval_env.step, in_axes=(0, 0, 0, None) # type: ignore
                    )(rng_step, eval_env_state, actions, eval_env_params)
                    done_batch=done
                    transitions=[]    

                    for i, train_state in enumerate(train_states):
                        done_batch['agents'][i] = batchify(done["agents"][i],config["NUM_ACTORS_PERTYPE"][i]).squeeze()
                        obs_batch = batchify(last_obs[i],config["NUM_ACTORS_PERTYPE"][i])
                        action_batch = batchify(actions[i],config["NUM_ACTORS_PERTYPE"][i])
                        value = values[i]
                        log_prob = log_probs[i]

                        info_i={"world":info["world"],"agent":jax.tree.map(lambda x: x.reshape(config["NUM_ACTORS_PERTYPE"][i],-1),info["agents"][i])}
                        # print(f"info for agenttype {i}:", info_i)


                        transitions.append(Transition(
                            jnp.tile(done["__all__"], config["NUM_AGENTS_PER_TYPE"][i]),
                            last_done[i],
                            action_batch.squeeze(),
                            value.squeeze(),
                            batchify(reward[i], config["NUM_ACTORS_PERTYPE"][i]).squeeze(),
                            log_prob.squeeze(),
                            obs_batch,
                            info_i,
                            # avail_actions,
                        ))
                    eval_runner_state = (train_states, eval_env_state, obsv, done_batch['agents'], h_states, rng)
                    return eval_runner_state, transitions

                rng, _rng = jax.random.split(rng)
                reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
                eval_obsv, eval_env_state = jax.vmap(eval_env.reset, in_axes=(0, None))(reset_rng, eval_env_params) # type: ignore


                eval_hstates=[]
                init_dones_agents_eval=[]
                for i, train_state in enumerate(train_states):
                    eval_hstates.append(ScannedRNN.initialize_carry(config["NUM_ACTORS_PERTYPE"][i], config["GRU_HIDDEN_DIM"]))
                    init_dones_agents_eval.append(jnp.zeros((config["NUM_ACTORS_PERTYPE"][i]), dtype=bool))


                
                eval_runner_state = (
                train_states,
                eval_env_state,
                eval_obsv,
                init_dones_agents_eval,
                eval_hstates,
                _rng,
                )
                eval_runner_state, eval_traj_batch = jax.lax.scan(
                    _eval_step, eval_runner_state, None,  config["NUM_STEPS_EVAL"]
                )
                metrics['agents_eval'] = [jax.tree.map(
                    lambda x: x.reshape(
                        (config["NUM_STEPS_EVAL"], config["NUM_ENVS"], config["NUM_AGENTS_PER_TYPE"][i])
                    ),
                    trjbtch.info['agent']) for i, trjbtch in enumerate(eval_traj_batch)]
                metrics['world_eval'] = [trjbtch.info['world'] for i, trjbtch in enumerate(eval_traj_batch)]
                if config["CALC_EVAL"]:
                    metrics['avg_reward_eval'] = [jnp.mean(tr.reward) for tr in eval_traj_batch]
                    metrics["traj_batch_eval"] = eval_traj_batch

            def callback(metric):
                print("Update step:", metric["update_steps"])
                # for loss_idx, m in enumerate(metric["loss"]):
                #         logging_dict[f"agent_{agent_name}/loss_{loss_idx}"] = m
                # Needed?

                for agent_index, tr in enumerate(metric["traj_batch"]):
                    agent_name = agent_type_names[agent_index]

                    action_distribution = {}
                    print("Action shape is ",tr.action.shape)
                    actions = np.array(tr.action).reshape(-1, *tr.action.shape[2:])
                    if actions.ndim>1:
                        avg_quant=np.mean(actions,axis=0)
                        for i,aq in enumerate(avg_quant):
                            action_distribution[f"agent_{agent_name}/action_dim_{i}_mean_quant"] = aq
                    else:
                        unique_actions, counts = np.unique(actions, return_counts=True)
                        tot_counts=sum(counts)
                        # Add each action count to the dictionary with a unique key
                        for a, c in zip(unique_actions, counts):
                            action_distribution[f"agent_{agent_name}/action_{int(a)}"] = c/tot_counts*100

                    # Extract slippage metrics from info dict
                    twap_slippage = None
                    vwap_slippage = None
                    twap_slippage_precision = None
                    twap_slippage_recall = None
                    vwap_slippage_precision = None
                    vwap_slippage_recall = None
                    
                    if 'twap_slippage' in tr.info['agent']:
                        twap_slippage_array = np.array(tr.info['agent']['twap_slippage'])
                        twap_slippage = float(np.mean(twap_slippage_array))
                        # Classify slippage: 1 if below median (good), 0 if above median (bad)
                        if len(twap_slippage_array) > 1:
                            slippage_median = np.median(twap_slippage_array)
                            twap_slippage_binary = (twap_slippage_array < slippage_median).astype(int)
                            # Create dummy predictions (in a real scenario, these would come from a model)
                            # For now, we'll create a reasonable prediction based on action
                            if 'pred_twap_slippage' in tr.info['agent']:
                                pred_array = np.array(tr.info['agent']['pred_twap_slippage'])
                                pred_binary = (pred_array < slippage_median).astype(int)
                                twap_slippage_precision, twap_slippage_recall = _compute_slippage_metrics(twap_slippage_binary, pred_binary)
                    
                    if 'vwap_slippage' in tr.info['agent']:
                        vwap_slippage_array = np.array(tr.info['agent']['vwap_slippage'])
                        vwap_slippage = float(np.mean(vwap_slippage_array))
                        # Classify slippage: 1 if below median (good), 0 if above median (bad)
                        if len(vwap_slippage_array) > 1:
                            slippage_median = np.median(vwap_slippage_array)
                            vwap_slippage_binary = (vwap_slippage_array < slippage_median).astype(int)
                            if 'pred_vwap_slippage' in tr.info['agent']:
                                pred_array = np.array(tr.info['agent']['pred_vwap_slippage'])
                                pred_binary = (pred_array < slippage_median).astype(int)
                                vwap_slippage_precision, vwap_slippage_recall = _compute_slippage_metrics(vwap_slippage_binary, pred_binary)

                    logging_dict = {
                        "env_step": (metric["update_steps"]+1)
                        * config["NUM_ENVS"]
                        * config["NUM_STEPS"],
                        **{f"agent_{agent_name}/{j}": m for j, m in metric["loss"][agent_index].items()},
                        **{f"agent_{agent_name}/reward": metric["avg_reward"][agent_index]},
                        **action_distribution,
                        **({f"agent_{agent_name}/twap_slippage": twap_slippage} if twap_slippage is not None else {}),
                        **({f"agent_{agent_name}/vwap_slippage": vwap_slippage} if vwap_slippage is not None else {}),
                        **({f"agent_{agent_name}/twap_slippage_precision": twap_slippage_precision} if twap_slippage_precision is not None else {}),
                        **({f"agent_{agent_name}/twap_slippage_recall": twap_slippage_recall} if twap_slippage_recall is not None else {}),
                        **({f"agent_{agent_name}/vwap_slippage_precision": vwap_slippage_precision} if vwap_slippage_precision is not None else {}),
                        **({f"agent_{agent_name}/vwap_slippage_recall": vwap_slippage_recall} if vwap_slippage_recall is not None else {}),
                    }
                
                    
                    for key, value in tr.info['agent'].items():
                    # Check if value is a numpy array or jax array and has elements
                        if isinstance(value, (jnp.ndarray, np.ndarray)) and value.size > 0:
                            flat_value = np.array(value).flatten()
                            if flat_value.size > 0:
                                # Get agent short_name from config
                                logging_dict[f"agent_{agent_name}/{key}_mean"] = float(np.mean(flat_value))
                                logging_dict[f"agent_{agent_name}/{key}_std"] = float(np.std(flat_value))
                    
                    # Process world info if available
                    if 'world' in tr.info and tr.info['world']:
                        for key, value in tr.info['world'].items():
                            if isinstance(value, (jnp.ndarray, np.ndarray)) and value.size > 0:
                                flat_value = np.array(value).flatten()
                                if flat_value.size > 0:
                                    logging_dict[f"world/{key}_mean"] = float(np.mean(flat_value))

                    # Add evaluation metrics if available
                    if config["CALC_EVAL"] and "traj_batch_eval" in metric:
                        tr= metric["traj_batch_eval"][agent_index]
                        agent_name = agent_type_names[agent_index]
                        action_distribution = {}
                        actions = np.array(tr.action).flatten()
                        unique_actions, counts = np.unique(actions, return_counts=True)
                        tot_counts=sum(counts)
                        # Add each action count to the dictionary with a unique key
                        for a, c in zip(unique_actions, counts):
                            action_distribution[f"eval_agent_{agent_name}/action_{int(a)}"] = c/tot_counts*100
                        logging_dict.update(action_distribution)
                        for key, value in tr.info['agent'].items():
                            if isinstance(value, (jnp.ndarray, np.ndarray)) and value.size > 0:
                                flat_value = np.array(value).flatten()
                                if flat_value.size > 0:
                                    logging_dict[f"eval_agent_{agent_name}/{key}_mean"] = float(np.mean(flat_value))
                                    logging_dict[f"eval_agent_{agent_name}/{key}_std"] = float(np.std(flat_value))
                        
                        # Process world eval info if available
                        if 'world' in tr.info and tr.info['world']:
                            for key, value in tr.info['world'].items():
                                if isinstance(value, (jnp.ndarray, np.ndarray)) and value.size > 0:
                                    flat_value = np.array(value).flatten()
                                    if flat_value.size > 0:
                                        logging_dict[f"eval_world/{key}_mean"] = float(np.mean(flat_value))

                        logging_dict.update({
                            **{f"eval_agent_{agent_name}/reward": metric["avg_reward_eval"][agent_index]},
                        })
                    if config["WANDB_MODE"]!= "disabled":
                        wandb.log(logging_dict)


                for agent_index, agent_value in enumerate(metric["avg_reward"]):
                    agent_name = agent_type_names[agent_index]
                    print(f"avg_reward_{agent_name} {agent_value}")

            metrics["update_steps"] = update_steps
            jax.experimental.io_callback(callback, None, metrics)
            update_steps = update_steps + 1
            runner_state = (train_states, env_state, last_obs, last_dones, hstates_new, rng)

            print("Finished compiling")
            # jax.profiler.save_device_memory_profile(f"memory_{update_steps}.prof")
            return (runner_state, update_steps), metrics

        rng, _rng = jax.random.split(rng)

        # Shard batch-dimension data across devices, replicate params/rng
        def shard_batch(x):
            return jax.device_put(x, batch_sharding)
        def replicate(x):
            return jax.device_put(x, replicated_sharding)

        env_state = jax.tree.map(shard_batch, env_state)
        obsv = jax.tree.map(shard_batch, obsv)
        hstates = [shard_batch(h) for h in hstates]
        init_dones_agents = [shard_batch(d) for d in init_dones_agents]

        train_states = [jax.tree.map(replicate, ts) for ts in train_states]
        env_params = jax.tree.map(replicate, env_params)
        if eval_env_params is not None:
            eval_env_params = jax.tree.map(replicate, eval_env_params)
        _rng = replicate(_rng)

        runner_state = (
            train_states,
            env_state,
            obsv,
            init_dones_agents, # last_done
            hstates,  # initial hidden states for RNN
            _rng,
        )

        jitted_update_step = jax.jit(_update_step)
        

        checkpoint_dir=f'{config["world_config"]["alphatradePath"]}/checkpoints/MARLCheckpoints/{config["PROJECT"]}/{(run.name if run.name else run.id) if run else "GENERIC_RUN"}'
        orbax_checkpointer = oxcp.PyTreeCheckpointer()
        options = oxcp.CheckpointManagerOptions(max_to_keep=2, create=True,keep_period=config["NUM_UPDATES"]//2)
        checkpoint_manager = oxcp.CheckpointManager(
             checkpoint_dir, orbax_checkpointer, options
                )
        print("Saving checkpoints to directory: \n \t",checkpoint_dir)


        
        updates=0
        for i in range(config["NUM_UPDATES"]):
            print(f"Starting Update step {i+1}/{config['NUM_UPDATES']}")
            # Run the update step:
            #if i>2 and i<4:
                #jax.profiler.start_trace("/tmp/profile-data")
            (runner_state,updates),metrics=jitted_update_step((runner_state,updates),env_params,eval_env_params,None)
            # if i>2 and i<4:
            #     jax.block_until_ready((runner_state,updates,metrics))
            #     jax.profiler.stop_trace()
            print(f"Update step {updates} completed")
            if i == 0 and n_devices > 1:
                sample_param = jax.tree.leaves(runner_state[0][0].params)[0]
                sample_obs = jax.tree.leaves(runner_state[2])[0]
                print(f"Param sharding: {sample_param.sharding}")
                print(f"Obs sharding: {sample_obs.sharding}")
            if config["CALC_EVAL"]:
                ckpt = {
                    'model': runner_state[0],  # train_states
                    # 'config': config if isinstance(config, dict) else config.as_dict(),
                    'metrics': {
                        'train_rewards': metrics["avg_reward"],
                        'eval_rewards': metrics["avg_reward_eval"],
                        }
                }
            else:
                ckpt = {
                    'model': runner_state[0],  # train_states
                    # 'config': config if isinstance(config, dict) else config.as_dict(),
                    'metrics': {
                        'train_rewards': metrics["avg_reward"],
                        }
                }
            print(f"Saving checkpoint {updates}")
            save_args = orbax_utils.save_args_from_target(ckpt)
            checkpoint_manager.save(updates, ckpt, save_kwargs={"save_args": save_args})
            del metrics
            gc.collect()
        

        checkpoint_manager.wait_until_finished()

        # runner_state, metrics = jax.lax.scan(
        #     _update_step, (runner_state, 0), None, config["NUM_UPDATES"]
        # )
        
        
        return {"runner_state": runner_state}

    return train


@hydra.main(version_base="1.3", config_path="../../../config/rl_configs", config_name="ippo_rnn_JAXMARL_exec")
def main(config):
    try:
        if config["ENV_CONFIG"] is not None:
            print(f"Loading the env config from file \n\t{config['ENV_CONFIG']} ")
            env_config=load_config_from_file(config["ENV_CONFIG"])
            print("********* DEBUG ********** \n Loaded env_config: ", env_config)
        else:
            print("Using default MultiAgentConfig as defined in jaxob_config.py file.")
            env_config=MultiAgentConfig()
            save_config_to_file(env_config,f"config/env_configs/default_config_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    except Exception as e:
        print(f"Error loading env config: {e}")
        print("Reverting to default MultiAgentConfig as defined in jaxob_config.py file.")
        env_config=MultiAgentConfig()
        save_config_to_file(env_config,f"config/env_configs/default_config_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    print("Note: The sweep parameters in yaml will override these settings.")
    env_config = OmegaConf.create(
        OmegaConf.to_container(OmegaConf.structured(env_config), resolve=True)
    )
    final_config=OmegaConf.merge(env_config, config)
    config = OmegaConf.to_container(final_config)


    # Ensure sweep_parameters is also properly converted
    sweep_parameters = config.get("SWEEP_PARAMETERS", {})
    if sweep_parameters is None:
        sweep_parameters = {}
    # Double-check it's a proper dict (in case of nested OmegaConf objects)
    if hasattr(sweep_parameters, '__dict__') and not isinstance(sweep_parameters, dict):
        sweep_parameters = OmegaConf.to_container(sweep_parameters, resolve=True)

    sweep_config={
        "method": "grid",
        "parameters": sweep_parameters,
    }

    # In sweep mode, SEED is excluded from base config so the sweep can provide it.
    # In single-run mode, SEED must be in the config directly.
    is_single_run = not sweep_parameters or config.get("WANDB_MODE") == "disabled"

    def sweep_fun():
        # print(f"WANDB CONFIG PRIOR {wandb.config}")
        run=wandb.init(
            entity=config["ENTITY"], # type: ignore
            project=config["PROJECT"], # type: ignore
            tags=["IPPO", "RNN", str(config.get("TRAINING_MODE", "rl_cold"))], # type: ignore
            config=config, # type: ignore
            mode=config["WANDB_MODE"], # type: ignore
            allow_val_change=True,
            config_exclude_keys=[] if is_single_run else ["SEED"],
        )

    
        # +++++ Single GPU +++++
        rng = jax.random.PRNGKey(wandb.config["SEED"])

        print("Final check: the wandb.config object used in this run is \n \t", wandb.config)

        if config["Timing"]:
            start_time = time.time()


        train_fun = make_train(wandb.config)
        out = train_fun(rng,run)
        # train_state = out['runner_state'][0] # runner_state.train_state
        # params = train_state.params

        if config["Timing"]:
            end_time = time.time()
            elapsed = end_time - start_time
            total_steps = config["TOTAL_TIMESTEPS"]
            agents_per_type = config["NUM_AGENTS_PER_TYPE"]
            num_data_msgs = config.get("n_data_msg_per_step", None)
            num_envs = config["NUM_ENVS"]

            # Print results
            print(f"Total steps: {total_steps}")
            print(f"Elapsed time: {elapsed} seconds")
            print(f"Steps per second: {total_steps / elapsed}")
            print(f"Agents per type: {agents_per_type}")
            print(f"Num data messages: {num_data_msgs}")
            print(f"Num envs: {num_envs}")

            # Save to CSV
            # Log timing metrics to wandb
            wandb.log({
                "timing/total_steps": total_steps,
                "timing/elapsed_seconds": elapsed,
                "timing/steps_per_second": total_steps / elapsed,
                "timing/agents_per_type": str(agents_per_type),
                "timing/num_data_msgs": num_data_msgs,
                "timing/num_envs": num_envs,
            })
            # -------------OBSOLETE TIMING SAVE - kept for reference --------------

            # df = pd.DataFrame(results)
            # csv_path = "timing_results.csv"
            # # Append if file exists, else write header
            # try:
            #     with open(csv_path, "x", newline="") as f:
            #         df.to_csv(f, index=False)
            # except FileExistsError:
            #     with open(csv_path, "a", newline="") as f:
            #         df.to_csv(f, index=False, header=False)

        # -------------OBSOLETE PARAMETER SAVE/LOAD CODE - kept for reference --------------
        # # Save the params to a file using flax.serialization.to_bytes
        # with open(params_file_name, 'wb') as f:
        #     f.write(flax.serialization.to_bytes(params))
        #     print(f"params saved")

        # Load the params from the file using flax.serialization.from_bytes
        # with open(params_file_name, 'rb') as f:
        #     restored_params = flax.serialization.from_bytes(flax.core.frozen_dict.FrozenDict, f.read())
        #     print(f"params restored")
        # Clean up resources after training
        del out
        gc.collect()

        # Force JAX to release memory
        jax.clear_caches()
        jax.local_devices()  # This can help trigger cleanup of device buffers
        run.finish()
    # If no sweep parameters or wandb disabled, run a single training directly
    if is_single_run:
        print("Running single training (no WandB sweep).")
        sweep_fun()
    else:
        sweep_id = wandb.sweep(sweep=sweep_config, project=config["PROJECT"],entity=config["ENTITY"])
        print("The sweep ID is: ",sweep_id)
        wandb.agent(sweep_id, function=sweep_fun, count=500,)


    sys.exit(0)



if __name__ == "__main__":
    main()
