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
"""Search policies."""
import functools
from typing import Optional, Tuple

import chex
import jax
import jax.numpy as jnp

from afn_mctx._src import action_selection
from afn_mctx._src import base
from afn_mctx._src import qtransforms
from afn_mctx._src import search
from afn_mctx._src import seq_halving


def gumbel_aflownet_policy(
    params: base.Params,
    rng_key: chex.PRNGKey,
    root: base.RootFnOutput,
    recurrent_fn: base.RecurrentFn,
    num_simulations: int,
    invalid_actions: Optional[chex.Array] = None,
    max_depth: Optional[int] = None,
    loop_fn: base.LoopFn = jax.lax.fori_loop,
    *,
    qtransform: base.QTransform = qtransforms.qtransform_by_completion,
    adversarial: bool = False,
    alpha: float = 1.0,
    omega: float = 1.0,
    epsilon: float = 0.1,
    max_num_considered_actions: int = 16,
    gumbel_scale: chex.Numeric = 1.,
) -> base.PolicyOutput[action_selection.GumbelMuZeroExtraData]:
  """Runs Gumbel search for AFlowNets and returns the `PolicyOutput`.

  Args:
    params: params to be forwarded to root and recurrent functions.
    rng_key: random number generator state, the key is consumed.
    root: a `(prior_logits, value, embedding)` `RootFnOutput`. The
      `prior_logits` are from a policy network. The shapes are
      `([B, num_actions], [B], [B, ...])`, respectively.
    recurrent_fn: a callable to be called on the leaf nodes and unvisited
      actions retrieved by the simulation step, which takes as args
      `(params, rng_key, action, embedding)` and returns a `RecurrentFnOutput`
      and the new state embedding. The `rng_key` argument is consumed.
    num_simulations: the number of simulations.
    invalid_actions: a mask with invalid actions. Invalid actions
      have ones, valid actions have zeros in the mask. Shape `[B, num_actions]`.
    max_depth: maximum search tree depth allowed during simulation.
    loop_fn: Function used to run the simulations. It may be required to pass
      hk.fori_loop if using this function inside a Haiku module.
    qtransform: function to obtain completed Q-values for a node.
    adversarial: Indicates whether this is an adversarial problem.
    alpha: The hyperparameter that dictates the "softness" of flow propagation.
    omega: The hyperparameter that dictates the temperature of flow propagation.
    epsilon: E2W exploration hyperparameter.
    max_num_considered_actions: the maximum number of actions expanded at the
      root node. A smaller number of actions will be expanded if the number of
      valid actions is smaller.
    gumbel_scale: scale for the Gumbel noise. Evalution on perfect-information
      games can use gumbel_scale=0.0.

  Returns:
    `PolicyOutput` containing the proposed action, action_weights and the used
    search tree.
  """
  # Masking invalid actions.
  root = root.replace(
      prior_logits=_mask_invalid_actions(root.prior_logits, invalid_actions))

  # Generating Gumbel.
  rng_key, gumbel_rng = jax.random.split(rng_key)
  gumbel = gumbel_scale * jax.random.gumbel(
      gumbel_rng, shape=root.prior_logits.shape, dtype=root.prior_logits.dtype)

  # Searching.
  extra_data = action_selection.GumbelMuZeroExtraData(root_gumbel=gumbel)
  # TODO: TESTING SOFT ACTIONS!!!
  search_tree = search.search(
      params=params,
      rng_key=rng_key,
      root=root,
      recurrent_fn=recurrent_fn,
      root_action_selection_fn=functools.partial(
          action_selection.soft_sampling_fn,
          alpha=alpha,
          epsilon=epsilon,
          qtransform=qtransform,
      ),
      interior_action_selection_fn=functools.partial(
          action_selection.soft_sampling_fn,
          alpha=alpha,
          epsilon=epsilon,
          qtransform=qtransform,
      ),
      num_simulations=num_simulations,
      max_depth=max_depth,
      invalid_actions=invalid_actions,
      extra_data=extra_data,
      loop_fn=loop_fn,
      adversarial=adversarial,
      alpha=alpha,
      omega=omega,
  )
  summary = search_tree.summary()

  # Acting with the best action from the most visited actions.
  # The "best" action has the highest `gumbel + logits + q`.
  # Inside the minibatch, the considered_visit can be different on states with
  # a smaller number of valid actions.
  considered_visit = jnp.max(summary.visit_counts, axis=-1, keepdims=True)
  # The completed_qvalues include imputed values for unvisited actions.
  completed_qvalues = jax.vmap(qtransform, in_axes=[0, None])(  # pytype: disable=wrong-arg-types  # numpy-scalars  # pylint: disable=line-too-long
      search_tree, search_tree.ROOT_INDEX)
  to_argmax = seq_halving.score_considered(
      considered_visit, gumbel, root.prior_logits, completed_qvalues,
      summary.visit_counts)
  action = action_selection.masked_argmax(to_argmax, invalid_actions)

  # Producing action_weights usable to train the policy network.
  completed_search_logits = _mask_invalid_actions(
      root.prior_logits + completed_qvalues, invalid_actions)
  # Corresponds to \pi^\alpha(s_0).
  action_weights = jax.nn.softmax(alpha * completed_search_logits)
  return base.PolicyOutput(
      action=action,
      action_weights=action_weights,
      search_tree=search_tree)


def _mask_invalid_actions(logits, invalid_actions):
  """Returns logits with zero mass to invalid actions."""
  if invalid_actions is None:
    return logits
  chex.assert_equal_shape([logits, invalid_actions])
  logits = logits - jnp.max(logits, axis=-1, keepdims=True)
  # At the end of an episode, all actions can be invalid. A softmax would then
  # produce NaNs, if using -inf for the logits. We avoid the NaNs by using
  # a finite `min_logit` for the invalid actions.
  min_logit = jnp.finfo(logits.dtype).min
  return jnp.where(invalid_actions, min_logit, logits)
