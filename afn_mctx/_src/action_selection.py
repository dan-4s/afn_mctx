# Copyright 2021 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""A collection of action selection functions."""
from typing import Optional, TypeVar

import chex
import jax
import jax.numpy as jnp

from afn_mctx._src import base
from afn_mctx._src import qtransforms
from afn_mctx._src import seq_halving
from afn_mctx._src import tree as tree_lib


def switching_action_selection_wrapper(
    root_action_selection_fn: base.RootActionSelectionFn,
    interior_action_selection_fn: base.InteriorActionSelectionFn
) -> base.InteriorActionSelectionFn:
  """Wraps root and interior action selection fns in a conditional statement."""

  def switching_action_selection_fn(
      rng_key: chex.PRNGKey,
      tree: tree_lib.Tree,
      node_index: base.NodeIndices,
      depth: base.Depth) -> chex.Array:
    return jax.lax.cond(
        depth == 0,
        lambda x: root_action_selection_fn(*x),
        lambda x: interior_action_selection_fn(*x),
        (rng_key, tree, node_index))

  return switching_action_selection_fn


def soft_sampling_fn(
      rng_key: chex.PRNGKey,
      tree: tree_lib.Tree,
      node_index: base.NodeIndices,
      *,
      alpha: float = 1.0,
      epsilon: float = 0.1,
      qtransform: base.QTransform = qtransforms.qtransform_by_completion,
) -> chex.Array:
  """
  Returns the stochastically selected action from the soft mellowmax policy
  function. 

  Q-values (state-action flow) are completed with the parent's flow prediction.

  Args:
    rng_key: random number generator state.
    tree: _unbatched_ MCTS tree state.
    node_index: scalar index of the node from which to select an action.
    depth: the scalar depth of the current node. The root has depth zero.
    alpha: spikyness hyperparameter for AFNs / mellowmax.
    adversarial: `bool` whether the environment is adversarial.
    epsilon: exploration constant for E2W.
    qtransform: a monotonic transformation of the Q-values.

  Returns:
    action: the stochastically sampled action following the soft mellowmax\
      policy.

  """
  rng_key, cat_key, noise_key = jax.random.split(rng_key, 3)
  chex.assert_shape([node_index], ())
  visit_counts = tree.children_visits[node_index]
  prior_logits = tree.children_prior_logits[node_index]
  chex.assert_equal_shape([visit_counts, prior_logits])
  completed_qvalues = qtransform(tree, node_index)
  invalid_actions = tree.invalid_actions[node_index]
  num_actions = tree.num_actions
  num_simulations = tree.num_simulations

  # Construct the soft mellowmax policy: exp(alpha * F) / sum(...). Then,
  # construct the uniform policy and the E2W policy from the MENTS paper.
  parent_policy = jax.nn.softmax(alpha * completed_qvalues)
  uniform_policy = jnp.ones_like(parent_policy) / num_actions
  lambda_t = epsilon * num_actions / jnp.log(num_simulations + 2)
  E2W_policy = (1 - lambda_t) * parent_policy + lambda_t * uniform_policy

  # Convert to log-space for the categorical choice later (should be log-stable,
  # since jnp.log(0) = -inf), however, may cause unexpected errors.
  E2W_logits = jnp.log(E2W_policy)
  
  # Mask invalid actions.
  E2W_logits = jnp.where(
      invalid_actions,
      -jnp.inf,
      E2W_logits,
  )

  # Sample from the constructed parent policy.
  actions = jax.random.categorical(key=cat_key, logits=E2W_logits, axis=-1)
  chex.assert_shape([actions], ())

  return actions


def muzero_action_selection(
    rng_key: chex.PRNGKey,
    tree: tree_lib.Tree,
    node_index: chex.Numeric,
    depth: chex.Numeric,
    *,
    pb_c_init: float = 1.25,
    pb_c_base: float = 19652.0,
    qtransform: base.QTransform = qtransforms.qtransform_by_parent_and_siblings,
) -> chex.Array:
  """Returns the action selected for a node index.

  See Appendix B in https://arxiv.org/pdf/1911.08265.pdf for more details.

  Args:
    rng_key: random number generator state.
    tree: _unbatched_ MCTS tree state.
    node_index: scalar index of the node from which to select an action.
    depth: the scalar depth of the current node. The root has depth zero.
    pb_c_init: constant c_1 in the PUCT formula.
    pb_c_base: constant c_2 in the PUCT formula.
    qtransform: a monotonic transformation to convert the Q-values to [0, 1].

  Returns:
    action: the action selected from the given node.
  """
  visit_counts = tree.children_visits[node_index]
  node_visit = tree.node_visits[node_index]
  pb_c = pb_c_init + jnp.log((node_visit + pb_c_base + 1.) / pb_c_base)
  prior_logits = tree.children_prior_logits[node_index]
  prior_probs = jax.nn.softmax(prior_logits)
  policy_score = jnp.sqrt(node_visit) * pb_c * prior_probs / (visit_counts + 1)
  chex.assert_shape([node_index, node_visit], ())
  chex.assert_equal_shape([prior_probs, visit_counts, policy_score])
  value_score = qtransform(tree, node_index)

  # Add tiny bit of randomness for tie break
  node_noise_score = 1e-7 * jax.random.uniform(
      rng_key, (tree.num_actions,))
  to_argmax = value_score + policy_score + node_noise_score

  # Masking the invalid actions at the root.
  return masked_argmax(to_argmax, tree.root_invalid_actions * (depth == 0))


@chex.dataclass(frozen=True)
class GumbelMuZeroExtraData:
  """Extra data for Gumbel MuZero search."""
  root_gumbel: chex.Array


GumbelMuZeroExtraDataType = TypeVar(  # pylint: disable=invalid-name
    "GumbelMuZeroExtraDataType", bound=GumbelMuZeroExtraData)


def gumbel_muzero_root_action_selection(
    rng_key: chex.PRNGKey,
    tree: tree_lib.Tree[GumbelMuZeroExtraDataType],
    node_index: chex.Numeric,
    *,
    num_simulations: chex.Numeric,
    max_num_considered_actions: chex.Numeric,
    qtransform: base.QTransform = qtransforms.qtransform_completed_by_mix_value,
) -> chex.Array:
  """Returns the action selected by Sequential Halving with Gumbel.

  Initially, we sample `max_num_considered_actions` actions without replacement.
  From these, the actions with the highest `gumbel + logits + qvalues` are
  visited first.

  Args:
    rng_key: random number generator state.
    tree: _unbatched_ MCTS tree state.
    node_index: scalar index of the node from which to take an action.
    num_simulations: the simulation budget.
    max_num_considered_actions: the number of actions sampled without
      replacement.
    qtransform: a monotonic transformation for the Q-values.

  Returns:
    action: the action selected from the given node.
  """
  del rng_key
  chex.assert_shape([node_index], ())
  visit_counts = tree.children_visits[node_index]
  prior_logits = tree.children_prior_logits[node_index]
  chex.assert_equal_shape([visit_counts, prior_logits])
  completed_qvalues = qtransform(tree, node_index)
  root_invalid_actions = tree.invalid_actions[node_index]

  table = jnp.array(seq_halving.get_table_of_considered_visits(
      max_num_considered_actions, num_simulations))
  num_valid_actions = jnp.sum(
      1 - root_invalid_actions, axis=-1).astype(jnp.int32)
  num_considered = jnp.minimum(
      max_num_considered_actions, num_valid_actions)
  chex.assert_shape(num_considered, ())
  # At the root, the simulation_index is equal to the sum of visit counts.
  simulation_index = jnp.sum(visit_counts, -1)
  chex.assert_shape(simulation_index, ())
  considered_visit = table[num_considered, simulation_index]
  chex.assert_shape(considered_visit, ())
  gumbel = tree.extra_data.root_gumbel
  to_argmax = seq_halving.score_considered(
      considered_visit, gumbel, prior_logits, completed_qvalues,
      visit_counts)

  # Masking the invalid actions at the root.
  return masked_argmax(to_argmax, root_invalid_actions)


def gumbel_muzero_interior_action_selection(
    rng_key: chex.PRNGKey,
    tree: tree_lib.Tree,
    node_index: chex.Numeric,
    depth: chex.Numeric,
    *,
    qtransform: base.QTransform = qtransforms.qtransform_completed_by_mix_value,
) -> chex.Array:
  """Selects the action with a deterministic action selection.

  The action is selected based on the visit counts to produce visitation
  frequencies similar to softmax(prior_logits + qvalues).

  Args:
    rng_key: random number generator state.
    tree: _unbatched_ MCTS tree state.
    node_index: scalar index of the node from which to take an action.
    depth: the scalar depth of the current node. The root has depth zero.
    qtransform: function to obtain completed Q-values for a node.

  Returns:
    action: the action selected from the given node.
  """
  del rng_key, depth
  chex.assert_shape([node_index], ())
  visit_counts = tree.children_visits[node_index]
  # TODO: Again, change the prior logits so that they are from the updated flow
  # estimates and not the QF estimates, which are out of date. We still need to
  # use children_values here, which is problematic with a low simulation count.
  prior_logits = tree.children_prior_logits[node_index]
  chex.assert_equal_shape([visit_counts, prior_logits])
  completed_qvalues = qtransform(tree, node_index)

  # The `prior_logits + completed_qvalues` provide an improved policy,
  # because the missing qvalues are replaced by v_{prior_logits}(node).
  to_argmax = _prepare_argmax_input(
      probs=jax.nn.softmax(prior_logits + completed_qvalues),
      visit_counts=visit_counts)

  chex.assert_rank(to_argmax, 1)
  return jnp.argmax(to_argmax, axis=-1).astype(jnp.int32)


def masked_argmax(
    to_argmax: chex.Array,
    invalid_actions: Optional[chex.Array]) -> chex.Array:
  """Returns a valid action with the highest `to_argmax`."""
  if invalid_actions is not None:
    chex.assert_equal_shape([to_argmax, invalid_actions])
    # The usage of the -inf inside the argmax does not lead to NaN.
    # Do not use -inf inside softmax, logsoftmax or cross-entropy.
    to_argmax = jnp.where(invalid_actions, -jnp.inf, to_argmax)
  # If all actions are invalid, the argmax returns action 0.
  return jnp.argmax(to_argmax, axis=-1).astype(jnp.int32)


def _prepare_argmax_input(probs, visit_counts):
  """Prepares the input for the deterministic selection.

  When calling argmax(_prepare_argmax_input(...)) multiple times
  with updated visit_counts, the produced visitation frequencies will
  approximate the probs.

  For the derivation, see Section 5 "Planning at non-root nodes" in
  "Policy improvement by planning with Gumbel":
  https://openreview.net/forum?id=bERaNdoegnO

  Args:
    probs: a policy or an improved policy. Shape `[num_actions]`.
    visit_counts: the existing visit counts. Shape `[num_actions]`.

  Returns:
    The input to an argmax. Shape `[num_actions]`.
  """
  chex.assert_equal_shape([probs, visit_counts])
  to_argmax = probs - visit_counts / (
      1 + jnp.sum(visit_counts, keepdims=True, axis=-1))
  return to_argmax
