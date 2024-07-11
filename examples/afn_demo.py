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
import jax.numpy as jnp
import mctx

FLAGS = flags.FLAGS
flags.DEFINE_integer("seed", 42, "Random seed.")
flags.DEFINE_integer("batch_size", 4, "Batch size.")
flags.DEFINE_integer("num_actions", 2, "Number of actions.")
flags.DEFINE_integer("num_simulations", 4, "Number of simulations.")
flags.DEFINE_integer("max_num_considered_actions", 2,
                     "The maximum number of actions expanded at the root.")
flags.DEFINE_integer("num_runs", 1, "Number of runs on random data.")


@chex.dataclass(frozen=True)
class DemoOutput:
  prior_policy_value: chex.Array
  prior_policy_action_value: chex.Array
  selected_action_value: chex.Array
  action_weights_policy_value: chex.Array


def _run_demo(rng_key: chex.PRNGKey) -> Tuple[chex.PRNGKey, DemoOutput]:
  """
  Runs a search algorithm on a 2-level binary tree using MCTS for AFNs.
  """
  batch_size = FLAGS.batch_size
  rng_key, logits_rng, q_rng, search_rng = jax.random.split(rng_key, 4)
  
  # The logits will all be random priors, but we likely won't use them for
  # anything major.
  prior_logits = jax.random.normal(
      logits_rng, shape=[batch_size, FLAGS.num_actions])
  
  # We now need to define the environment that we're working with. 
  
  # Defining a bandit with random Q-values. Only the Q-values of the visited
  # actions will be revealed to the search algorithm.
  qvalues = jax.random.uniform(q_rng, shape=prior_logits.shape)
  # If we know the value under the prior policy, we can use the value to
  # complete the missing Q-values. The completed Q-values will produce an
  # improved policy in `policy_output.action_weights`.
  raw_value = jnp.sum(jax.nn.softmax(prior_logits) * qvalues, axis=-1)
  use_mixed_value = False

  # The root output would be the output of MuZero representation network.
  root = mctx.RootFnOutput(
      prior_logits=prior_logits,
      value=raw_value,
      # The embedding is used only to implement the MuZero model.
      embedding=jnp.zeros([batch_size]),
  )
  # The recurrent_fn would be provided by MuZero dynamics network.
  recurrent_fn = _make_bandit_recurrent_fn(qvalues)

  # Running the search.
  policy_output = mctx.gumbel_muzero_policy(
      params=(),
      rng_key=search_rng,
      root=root,
      recurrent_fn=recurrent_fn,
      num_simulations=FLAGS.num_simulations,
      max_num_considered_actions=FLAGS.max_num_considered_actions,
      qtransform=functools.partial(
          mctx.qtransform_completed_by_mix_value,
          use_mixed_value=use_mixed_value),
  )

  # Collecting the Q-value of the selected action.
  selected_action_value = qvalues[jnp.arange(batch_size), policy_output.action]

  # TODO: return some useful shit as output.
  output = None

  return rng_key, output


def _make_bandit_recurrent_fn(leaf_rewards):
  """
    Returns a recurrent_fn for an AFN on a 2-level binary tree.
  """

  def recurrent_fn(params, rng_key, action, embedding):
    del params, rng_key
    # Zero reward until the terminal nodes
    reward = jnp.where(embedding == 2,
                       leaf_rewards[jnp.arange(action.shape[0]), action],
                       0.0)
    # TODO: Don't need a discount in our code, just need to reimplement backprop.
    discount = jnp.ones_like(reward)
    recurrent_fn_output = mctx.RecurrentFnOutput(
        reward=reward,
        discount=discount,
        prior_logits=jnp.zeros_like(reward),
        value=jnp.zeros_like(reward))
    next_embedding = embedding + 1
    return recurrent_fn_output, next_embedding

  return recurrent_fn


def main(_):
  rng_key = jax.random.PRNGKey(FLAGS.seed)
  jitted_run_demo = jax.jit(_run_demo)
  for _ in range(FLAGS.num_runs):
    rng_key, output = jitted_run_demo(rng_key)
    
    # Print the flow and root action probabilities.
    # TODO


if __name__ == "__main__":
  app.run(main)



