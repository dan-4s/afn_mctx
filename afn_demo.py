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
import matplotlib.pyplot as plt
import afn_mctx
from examples.binary_env import Game, GameState

FLAGS = flags.FLAGS
flags.DEFINE_integer("seed", 42, "Random seed.")
flags.DEFINE_integer("batch_size", 4, "Batch size.")
flags.DEFINE_integer("num_actions", 2, "Number of actions.")
flags.DEFINE_integer("num_simulations", 4, "Number of simulations.")
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
  # of the root node. Shape = [B].
  root_value = jnp.sum(jax.nn.softmax(prior_logits) * qvalues, axis=-1)

  # The root output is used in the search tree.
  root = afn_mctx.RootFnOutput(
      prior_logits=prior_logits,
      value=root_value,
      # The embedding is any environment state information.
      embedding=states,
  )

  # The recurrent_fn takes care of environmental interactions estimating policy
  # and value priors if these are used in the algorithms.
  recurrent_fn = _make_alphazero_recurrent_fn(game_obj, num_actions)

  # Running the search.
  policy_output = afn_mctx.gumbel_muzero_policy(
      params=(),
      rng_key=search_rng,
      root=root,
      recurrent_fn=recurrent_fn,
      num_simulations=num_simulations,
      max_num_considered_actions=max_num_considered_actions,
      max_depth=2,
      qtransform=functools.partial(
          afn_mctx.qtransform_completed_by_mix_value,
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
    recurrent_fn_output = afn_mctx.RecurrentFnOutput(
        reward=reward,
        discount=discount,
        prior_logits=predicted_logits,
        value=predicted_value,
    )
    return recurrent_fn_output, states

  return recurrent_fn


def _run_aflownet_demo(
    rng_key: chex.PRNGKey,
    num_simulations,
    priors_method,
    noise_level,
    backward_method,
  ) -> Tuple[chex.PRNGKey, DemoOutput]:
  """
  Runs a search algorithm on a 2-level binary tree using MCTS for AFlowNets.
  This demo will follow the Gumbel implementation but is adapted for AFNs.
  """
  # Get all the flag values.
  batch_size = FLAGS.batch_size
  num_actions = FLAGS.num_actions
  # num_simulations = FLAGS.num_simulations
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
  # log_flows = jax.random.uniform(q_rng, shape=[batch_size, num_actions])
  if(priors_method == "random"):
    log_flows = jax.random.normal(logits_rng, shape=[batch_size, 2])
  else:
    log_flows = jnp.log(jnp.array([[0.1089, 1.089]] * batch_size))
  prior_logits = log_flows # These can be log flows, a softmax, or log softmax.

  # Use the prior policy and q-value estimates to generate the value estimate
  # of the root node. TODO: Can be done differently, will investigate later.
  log_root_flow = (logsumexp((alpha + 1)*log_flows, axis=1) -
                   logsumexp(alpha*log_flows, axis=1))

  # The root output is used in the search tree.
  root = afn_mctx.RootFnOutput(
      prior_logits=prior_logits,
      value=log_root_flow,
      # The embedding is any environment state information.
      embedding=states,
  )

  # The recurrent_fn takes care of environmental interactions estimating policy
  # and value priors if these are used in the algorithms.
  recurrent_fn = _make_afn_recurrent_fn(game_obj, num_actions, priors_method, noise_level)

  # Running the search.
  policy_output = afn_mctx.gumbel_aflownet_policy(
      params=(),
      rng_key=search_rng,
      root=root,
      recurrent_fn=recurrent_fn,
      num_simulations=num_simulations,
      max_num_considered_actions=max_num_considered_actions,
      max_depth=2,
      qtransform=functools.partial(
          afn_mctx.qtransform_completed_by_mix_value,
          use_mixed_value=False),
      backward_method=backward_method, # AFN or AFN_CONST -> AFN doesn't predict the QF constant.
  )

  return rng_key, policy_output


def _make_afn_recurrent_fn(game_obj: Game, num_actions: int, priors_method: str, noise_level: float):
  """
    Returns a recurrent_fn for an AlphaZero on a 2-level binary tree.
  """

  def recurrent_fn(params, rng_key, action, states: GameState):
    del params
    # Extract the rewards from the environment after taking an action. Rewards
    # are for AFNs this time, so we have to convert them to log-space.
    states = jax.vmap(game_obj.step)(state=states, action=action)
    reward = jax.vmap(game_obj.rewards, in_axes=[0, None])(states, True)
    log_reward = jnp.log(reward)
    reward = jnp.where(reward == 0, 0.0, log_reward)
    terminated = states.game_over

    # NOTE: Invert rewards to achieve correct operation! This is because the
    #       rewards are technically from the previous player's point of view!
    reward = -reward
    
    if(priors_method == "ground_truth" or priors_method == "mixed"):
      # Set the ground truth probabilities. (I know this is ugly, just bear with me).
      def generate_QF(board_states: chex.Array):
        return jnp.select(
          condlist=[board_states == 0, board_states == 1, board_states == 2],
          choicelist=[jnp.array([0.1089, 1.089]), jnp.array([10, 1]), jnp.array([1, 0.1])],
          default=0.5,
        ) 
      predicted_QF_gt = jax.vmap(generate_QF)(states.board)
      predicted_QF_gt = jnp.log(predicted_QF_gt)

      # Set the ground truth flow values.
      def generate_parent_flow(board_states: chex.Array):
        return jnp.select(
          condlist=[board_states == 0, board_states == 1, board_states == 2],
          choicelist=[1, 0.1089, 1.089],
          default=1.0,
        )
      predicted_flow_gt = jax.vmap(generate_parent_flow)(states.board)
      predicted_flow_gt = jnp.log(predicted_flow_gt)
    
    if(priors_method == "random" or priors_method == "mixed"):
      # Predict logits and values using RNG.
      batch_size = reward.shape[0]
      rng_key, QF_rng_key = jax.random.split(rng_key, 2)
      rng_key, flow_rng_key = jax.random.split(rng_key, 2)
      predicted_QF_r = jax.random.normal(QF_rng_key, shape=(batch_size, num_actions))
      predicted_flow_r = jax.random.normal(flow_rng_key, shape=(batch_size,))
    
    if(priors_method == "mixed"):
      predicted_flow = predicted_flow_gt + noise_level * predicted_flow_r
      predicted_QF = predicted_QF_gt + noise_level * predicted_QF_r
    elif(priors_method == "ground_truth"):
      predicted_flow = predicted_flow_gt
      predicted_QF = predicted_QF_gt
    else:
      predicted_flow = predicted_flow_r
      predicted_QF = predicted_QF_r
    
    predicted_flow = jnp.where(terminated, reward, predicted_flow)

    # Since this is AlphaZero-like in a two-player environment, we want to
    # apply a negation to the reward as a discount. If we're at a terminal
    # state, then the discount is zero. If we're at a non-terminal state, the
    # discount is -1.
    discount = -1.0 * jnp.ones_like(reward)
    discount = jnp.where(terminated, 0.0, discount)
    recurrent_fn_output = afn_mctx.RecurrentFnOutput(
        reward=reward, # Provide reward, but don't use it since it is already included in the flow.
        discount=discount,
        prior_logits=predicted_QF,
        value=predicted_flow,
    )
    return recurrent_fn_output, states

  return recurrent_fn


def main(_):
  rng_key = jax.random.PRNGKey(FLAGS.seed) # TODO: Change the seed.
  jitted_run_demo = jax.jit(_run_gumbel_alphazero_demo)
  # print("\n================================")
  # print("\tAlphaZero demos")
  # for i in range(FLAGS.num_runs):
  #   rng_key, policy_output = jitted_run_demo(rng_key)

  #   avg_action_weights = jnp.average(policy_output.action_weights, axis=0)
  #   print(f"Run {i+1} results")
  #   print(f"\tP(s1 | s0) = {avg_action_weights[0]:.2f}")
  #   print(f"\tP(s2 | s0) = {avg_action_weights[1]:.2f}")
  
  print("\n============================")
  print("\tAFN demos")
  jitted_run_demo = jax.jit(_run_aflownet_demo, static_argnums=[1,2,4])
  num_sims = [1, 2, 3, 4, 8, 50, 100, 1000]
  # num_sims = [1000] # TODO: TESTING!!
  # noise_schedule = [0.0] # TODO: TESTING!!
  noise_schedule = jnp.arange(start=0, stop=2.1, step=0.2)
  gt_flows = jnp.array([1.0, 11/101, 110/101])
  gt_log_flows = jnp.log(gt_flows)
  sims_to_errors = {}
  sims_to_KLs = {}
  for sims in num_sims:
    all_errors = []
    all_KLs = []
    for i in range(len(noise_schedule)):
      #We'll reuse the same rng_key for all experiments.
      _, policy_output = jitted_run_demo(rng_key, sims, "mixed", noise_schedule[i], "AFN_CONST")

      # Compute the error on the estimated flows.
      tree = policy_output.search_tree
      root_error = jnp.exp(tree.node_values[:,0]) - gt_flows[0]
      root_error = jnp.reshape(root_error, shape=(FLAGS.batch_size,1))
      child_error = jnp.exp(tree.children_values[:, 0]) - gt_flows[1:]
      total_error = jnp.concat((root_error, child_error), axis=1)
      average_error = jnp.average(total_error).item()
      all_errors.append(average_error)

      # Compute the KL divergence of the root policy.
      gt_policy = jax.nn.softmax(gt_log_flows[1:])
      mcts_policy = jax.nn.softmax(tree.children_values[:, 0, :])
      policy_div = jax.scipy.special.kl_div(gt_policy, mcts_policy)
      avg_policy_div = jnp.average(policy_div)
      all_KLs.append(avg_policy_div)

      # breakpoint()
    sims_to_errors[sims] = all_errors
    sims_to_KLs[sims] = all_KLs
  
  # Print graphs of the KL and error over the noise schedule.
  fig = plt.figure(figsize=(8,6))
  plt.plot(noise_schedule, sims_to_errors[50], "-x", label="50 sims")
  plt.plot(noise_schedule, sims_to_errors[100], "-o", label="100 sims")
  plt.plot(noise_schedule, sims_to_errors[1000], "-^", label="1000 sims")
  plt.title("Average error on flow, varying noise schedules")
  plt.ylabel("Average error")
  plt.xlabel("Noise level")
  plt.grid()
  plt.legend()
  plt.savefig("avg_error.png")
  plt.close()

  fig = plt.figure(figsize=(8,6))
  plt.plot(noise_schedule, sims_to_KLs[50], "-x", label="50 sims")
  plt.plot(noise_schedule, sims_to_KLs[100], "-o", label="100 sims")
  plt.plot(noise_schedule, sims_to_KLs[1000], "-^", label="1000 sims")
  plt.title("Average KL divergence of policies, varying noise levels")
  plt.ylabel("Average KL divergence")
  plt.xlabel("Noise level")
  plt.grid()
  plt.legend()
  plt.savefig("avg_kl.png")
  plt.close()


if __name__ == "__main__":
  app.run(main)



