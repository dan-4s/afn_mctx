"""
Demonstration of MCTS for AFNs as implemented by our new code.
"""

import functools
from typing import NamedTuple, Tuple

from absl import app
from absl import flags
import chex
import jax
from jax import Array
from jax.scipy.special import logsumexp
import jax.numpy as jnp
from .. import mctx
from binary_env import Game, GameState

FLAGS = flags.FLAGS
flags.DEFINE_integer("seed", 42, "Random seed.")
flags.DEFINE_integer("batch_size", 4, "Batch size.")
flags.DEFINE_integer("num_actions", 2, "Number of actions.")
flags.DEFINE_integer("num_simulations", 100, "Number of simulations.")
flags.DEFINE_integer("max_num_considered_actions", 2,
                     "The maximum number of actions expanded at the root.")
flags.DEFINE_integer("num_runs", 1, "Number of runs on random data.")
flags.DEFINE_integer("alpha", 1, "The exponentiation factor for generalised TB.")


@chex.dataclass(frozen=True)
class DemoOutput:
  prior_policy_value: chex.Array
  prior_policy_action_value: chex.Array
  selected_action_value: chex.Array
  action_weights_policy_value: chex.Array


def _run_gumbel_alphazero_demo(
    rng_key: chex.PRNGKey,
  ) -> Tuple[chex.PRNGKey, DemoOutput]:
  """
  Runs a search algorithm on a 2-level binary tree using Gumbel MCTS for
  AlphaZero.
  """
  # Get all the flag values.
  batch_size = FLAGS.batch_size
  num_actions = FLAGS.num_actions
  num_simulations = FLAGS.num_simulations
  max_num_considered_actions = FLAGS.max_num_considered_actions

  # Get more keys
  rng_key, logits_rng, q_rng, search_rng, env_key = jax.random.split(rng_key, 5)

  # Define the environment.
  env_keys = jax.random.split(env_key, batch_size)
  game_obj = Game()
  states = jax.vmap(Game.init)(env_keys)

  # The prior logits will all be random priors.
  prior_logits = jax.random.normal(
      logits_rng, shape=[batch_size, num_actions])
  
  # The prior q-value estimates will be random as well (no NN).
  qvalues = jax.random.uniform(q_rng, shape=prior_logits.shape)

  # Use the prior policy and q-value estimates to generate the value estimate
  # of the root node.
  root_value = jnp.sum(jax.nn.softmax(prior_logits) * qvalues, axis=-1)

  # The root output is used in the search tree.
  root = mctx.RootFnOutput(
      prior_logits=prior_logits,
      value=root_value,
      # The embedding is any environment state information.
      embedding=states,
  )

  # The recurrent_fn takes care of environmental interactions estimating policy
  # and value priors if these are used in the algorithms.
  recurrent_fn = _make_alphazero_recurrent_fn(game_obj, num_actions)

  # Running the search.
  policy_output = mctx.gumbel_muzero_policy(
      params=(),
      rng_key=search_rng,
      root=root,
      recurrent_fn=recurrent_fn,
      num_simulations=num_simulations,
      max_num_considered_actions=max_num_considered_actions,
      max_depth=2,
      qtransform=functools.partial(
          mctx.qtransform_completed_by_mix_value,
          use_mixed_value=False),
  )

  return rng_key, policy_output


def _make_alphazero_recurrent_fn(game_obj: Game, num_actions: int):
  """
    Returns a recurrent_fn for an AlphaZero on a 2-level binary tree.
  """

  def recurrent_fn(params, rng_key, action, states: GameState):
    del params
    # Extract the rewards from the environment after taking an action.
    states = jax.vmap(game_obj.step)(state=states, action=action)
    reward = jax.vmap(game_obj.rewards, in_axes=[0, None])(states, False)
    terminated = states.game_over

    # NOTE: Invert rewards to achieve correct operation! This is because the
    #       rewards are technically from the previous player's point of view!
    reward = -reward
    
    # Predict logits and values using RNG.
    batch_size = reward.shape[0]
    rng_key, policy_rng_key = jax.random.split(rng_key, 2)
    rng_key, value_rng_key = jax.random.split(rng_key, 2)
    predicted_logits = jax.random.uniform(policy_rng_key, (batch_size, num_actions))
    predicted_value = jax.random.uniform(value_rng_key, (batch_size,))
    predicted_value = jnp.where(terminated, 0.0, predicted_value)

    # Since this is AlphaZero-like in a two-player environment, we want to
    # apply a negation to the reward as a discount. If we're at a terminal
    # state, then the discount is zero. If we're at a non-terminal state, the
    # discount is -1.
    discount = -1.0 * jnp.ones_like(reward)
    discount = jnp.where(terminated, 0.0, discount)
    recurrent_fn_output = mctx.RecurrentFnOutput(
        reward=reward,
        discount=discount,
        prior_logits=predicted_logits,
        value=predicted_value,
    )
    return recurrent_fn_output, states

  return recurrent_fn


def _run_aflownet_demo(
    rng_key: chex.PRNGKey,
  ) -> Tuple[chex.PRNGKey, DemoOutput]:
  """
  Runs a search algorithm on a 2-level binary tree using MCTS for AFlowNets.
  This demo will follow the Gumbel implementation but is adapted for AFNs.
  """
  # Get all the flag values.
  batch_size = FLAGS.batch_size
  num_actions = FLAGS.num_actions
  num_simulations = FLAGS.num_simulations
  max_num_considered_actions = FLAGS.max_num_considered_actions
  alpha = FLAGS.alpha

  # Get more keys
  rng_key, logits_rng, q_rng, search_rng, env_key = jax.random.split(rng_key, 5)

  # Define the environment.
  env_keys = jax.random.split(env_key, batch_size)
  game_obj = Game()
  states = jax.vmap(Game.init)(env_keys)

  # The prior flow and policy estimates will be random (no NN). log_flows is
  # technically QF here, but we will not worry about exact estimations here.
  log_flows = jax.random.uniform(q_rng, shape=[batch_size, num_actions])
  prior_logits = log_flows # These can be log flows, a softmax, or log softmax.

  # Use the prior policy and q-value estimates to generate the value estimate
  # of the root node. TODO: Can be done differently, will investigate later.
  log_root_flow = logsumexp((alpha + 1)*log_flows) - logsumexp(alpha*log_flows)

  # The root output is used in the search tree.
  root = mctx.RootFnOutput(
      prior_logits=prior_logits,
      value=log_root_flow,
      # The embedding is any environment state information.
      embedding=states,
  )

  # The recurrent_fn takes care of environmental interactions estimating policy
  # and value priors if these are used in the algorithms.
  recurrent_fn = _make_afn_recurrent_fn(game_obj, num_actions, alpha)

  # Running the search.
  policy_output = mctx.gumbel_aflownet_policy(
      params=(),
      rng_key=search_rng,
      root=root,
      recurrent_fn=recurrent_fn,
      num_simulations=num_simulations,
      max_num_considered_actions=max_num_considered_actions,
      max_depth=2,
      qtransform=functools.partial(
          mctx.qtransform_completed_by_mix_value,
          use_mixed_value=False),
  )

  return rng_key, policy_output


def _make_afn_recurrent_fn(game_obj: Game, num_actions: int, alpha: float):
  """
    Returns a recurrent_fn for an AlphaZero on a 2-level binary tree.
  """

  def recurrent_fn(params, rng_key, action, states: GameState):
    del params
    # Extract the rewards from the environment after taking an action. Rewards
    # are for AFNs this time, so we have to convert them to log-space.
    states = jax.vmap(game_obj.step)(state=states, action=action)
    reward = jax.vmap(game_obj.rewards, in_axes=[0, None])(states, True)
    reward = jnp.log(reward)
    terminated = states.game_over

    # NOTE: Invert rewards to achieve correct operation! This is because the
    #       rewards are technically from the previous player's point of view!
    reward = -reward
    
    # Predict logits and values using RNG.
    batch_size = reward.shape[0]
    rng_key, QF_rng_key = jax.random.split(rng_key, 2)
    rng_key, flow_rng_key = jax.random.split(rng_key, 2)
    predicted_QF = jax.random.uniform(QF_rng_key, (batch_size, num_actions))
    predicted_flow = jax.random.uniform(flow_rng_key, (batch_size,))
    predicted_flow = jnp.where(terminated, reward, predicted_flow)

    # Since this is AlphaZero-like in a two-player environment, we want to
    # apply a negation to the reward as a discount. If we're at a terminal
    # state, then the discount is zero. If we're at a non-terminal state, the
    # discount is -1.
    discount = -1.0 * jnp.ones_like(reward)
    discount = jnp.where(terminated, 0.0, discount)
    recurrent_fn_output = mctx.RecurrentFnOutput(
        reward=reward, # Provide reward, but don't use it since it is already included in the flow.
        discount=discount,
        prior_logits=predicted_QF,
        value=predicted_flow,
    )
    return recurrent_fn_output, states

  return recurrent_fn


def main(_):
  rng_key = jax.random.PRNGKey(FLAGS.seed)
  jitted_run_demo = jax.jit(_run_gumbel_alphazero_demo)
  print("\n================================")
  print("\tAlphaZero demos")
  for i in range(FLAGS.num_runs):
    rng_key, policy_output = jitted_run_demo(rng_key)

    avg_action_weights = jnp.average(policy_output.action_weights, axis=0)
    print(f"Run {i+1} results")
    print(f"\tP(s1 | s0) = {avg_action_weights[0]:.2f}")
    print(f"\tP(s2 | s0) = {avg_action_weights[1]:.2f}")
  
  print("\n============================")
  print("\tAFN demos")
  jitted_run_demo = jax.jit(_run_aflownet_demo)
  for i in range(FLAGS.num_runs):
    rng_key, policy_output = jitted_run_demo(rng_key)

    avg_action_weights = jnp.average(policy_output.action_weights, axis=0)
    print(f"Run {i+1} results")
    print(f"\tP(s1 | s0) = {avg_action_weights[0]:.2f}")
    print(f"\tP(s2 | s0) = {avg_action_weights[1]:.2f}")
    breakpoint()


if __name__ == "__main__":
  app.run(main)



