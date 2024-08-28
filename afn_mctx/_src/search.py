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
"""A JAX implementation of batched MCTS."""
import functools
from typing import Any, NamedTuple, Optional, Tuple, TypeVar

import chex
import jax
import jax.numpy as jnp
import jax.scipy as jsp

from afn_mctx._src import action_selection
from afn_mctx._src import base
from afn_mctx._src import tree as tree_lib

Tree = tree_lib.Tree
T = TypeVar("T")


def search(
    params: base.Params,
    rng_key: chex.PRNGKey,
    *,
    root: base.RootFnOutput,
    recurrent_fn: base.RecurrentFn,
    root_action_selection_fn: base.RootActionSelectionFn,
    interior_action_selection_fn: base.InteriorActionSelectionFn,
    num_simulations: int,
    max_depth: Optional[int] = None,
    invalid_actions: Optional[chex.Array] = None,
    extra_data: Any = None,
    loop_fn: base.LoopFn = jax.lax.fori_loop,
    backward_method: Optional[str] = None,
    alpha: Optional[float] = 1.0) -> Tree:
  """Performs a full search and returns sampled actions.

  In the shape descriptions, `B` denotes the batch dimension.

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
    root_action_selection_fn: function used to select an action at the root.
    interior_action_selection_fn: function used to select an action during
      simulation.
    num_simulations: the number of simulations.
    max_depth: maximum search tree depth allowed during simulation, defined as
      the number of edges from the root to a leaf node.
    invalid_actions: a mask with invalid actions at the root. In the
      mask, invalid actions have ones, and valid actions have zeros.
      Shape `[B, num_actions]`.
    extra_data: extra data passed to `tree.extra_data`. Shape `[B, ...]`.
    loop_fn: Function used to run the simulations. It may be required to pass
      hk.fori_loop if using this function inside a Haiku module.
    backward_method: String which specifies the method for backwards flow
      propagation. Options: `None` for classic Gumbel method and 'AFN'.
    alpha: Number to use for the exponentiated flow.

  Returns:
    `SearchResults` containing outcomes of the search, e.g. `visit_counts`
    `[B, num_actions]`.
  """
  action_selection_fn = action_selection.switching_action_selection_wrapper(
      root_action_selection_fn=root_action_selection_fn,
      interior_action_selection_fn=interior_action_selection_fn
  )

  # Do simulation, expansion, and backward steps.
  batch_size = root.value.shape[0]
  batch_range = jnp.arange(batch_size)
  if max_depth is None:
    max_depth = num_simulations
  if invalid_actions is None:
    invalid_actions = jnp.zeros_like(root.prior_logits, dtype=jnp.bool)

  def body_fun(sim, loop_state):
    rng_key, tree = loop_state
    rng_key, simulate_key, expand_key = jax.random.split(rng_key, 3)
    # simulate is vmapped and expects batched rng keys.
    simulate_keys = jax.random.split(simulate_key, batch_size)
    parent_index, action = simulate(
        simulate_keys, tree, action_selection_fn, max_depth)
    # A node first expanded on simulation `i`, will have node index `i`.
    # Node 0 corresponds to the root node.
    next_node_index = tree.children_index[batch_range, parent_index, action]
    next_node_index = jnp.where(next_node_index == Tree.UNVISITED,
                                sim + 1, next_node_index)
    tree = expand(
        params, expand_key, tree, recurrent_fn, parent_index,
        action, next_node_index)
    tree = backward(tree, next_node_index, backward_method, alpha)
    loop_state = rng_key, tree
    return loop_state

  # Allocate all necessary storage.
  tree = instantiate_tree_from_root(root, num_simulations,
                                    root_invalid_actions=invalid_actions,
                                    extra_data=extra_data)
  _, tree = loop_fn(
      0, num_simulations, body_fun, (rng_key, tree))

  return tree


class _SimulationState(NamedTuple):
  """The state for the simulation while loop."""
  rng_key: chex.PRNGKey
  node_index: int
  action: int
  next_node_index: int
  depth: int
  is_continuing: bool


@functools.partial(jax.vmap, in_axes=[0, 0, None, None], out_axes=0)
def simulate(
    rng_key: chex.PRNGKey,
    tree: Tree,
    action_selection_fn: base.InteriorActionSelectionFn,
    max_depth: int) -> Tuple[chex.Array, chex.Array]:
  """Traverses the tree until reaching an unvisited action or `max_depth`.

  Each simulation starts from the root and keeps selecting actions traversing
  the tree until a leaf or `max_depth` is reached.

  Args:
    rng_key: random number generator state, the key is consumed.
    tree: _unbatched_ MCTS tree state.
    action_selection_fn: function used to select an action during simulation.
    max_depth: maximum search tree depth allowed during simulation.

  Returns:
    `(parent_index, action)` tuple, where `parent_index` is the index of the
    node reached at the end of the simulation, and the `action` is the action to
    evaluate from the `parent_index`.
  """
  def cond_fun(state):
    return state.is_continuing

  def body_fun(state):
    # Preparing the next simulation state.
    node_index = state.next_node_index
    rng_key, action_selection_key = jax.random.split(state.rng_key)
    action = action_selection_fn(action_selection_key, tree, node_index,
                                 state.depth)
    next_node_index = tree.children_index[node_index, action]
    # The returned action will be visited.
    depth = state.depth + 1
    is_before_depth_cutoff = depth < max_depth
    is_visited = next_node_index != Tree.UNVISITED
    is_continuing = jnp.logical_and(is_visited, is_before_depth_cutoff)
    return _SimulationState(  # pytype: disable=wrong-arg-types  # jax-types
        rng_key=rng_key,
        node_index=node_index,
        action=action,
        next_node_index=next_node_index,
        depth=depth,
        is_continuing=is_continuing)

  node_index = jnp.array(Tree.ROOT_INDEX, dtype=jnp.int32)
  depth = jnp.zeros((), dtype=tree.children_prior_logits.dtype)
  # pytype: disable=wrong-arg-types  # jnp-type
  initial_state = _SimulationState(
      rng_key=rng_key,
      node_index=tree.NO_PARENT,
      action=tree.NO_PARENT,
      next_node_index=node_index,
      depth=depth,
      is_continuing=jnp.array(True))
  # pytype: enable=wrong-arg-types
  end_state = jax.lax.while_loop(cond_fun, body_fun, initial_state)

  # Returning a node with a selected action.
  # The action can be already visited, if the max_depth is reached.
  return end_state.node_index, end_state.action


def expand(
    params: chex.Array,
    rng_key: chex.PRNGKey,
    tree: Tree[T],
    recurrent_fn: base.RecurrentFn,
    parent_index: chex.Array,
    action: chex.Array,
    next_node_index: chex.Array) -> Tree[T]:
  """Create and evaluate child nodes from given nodes and unvisited actions.

  Args:
    params: params to be forwarded to recurrent function.
    rng_key: random number generator state.
    tree: the MCTS tree state to update.
    recurrent_fn: a callable to be called on the leaf nodes and unvisited
      actions retrieved by the simulation step, which takes as args
      `(params, rng_key, action, embedding)` and returns a `RecurrentFnOutput`
      and the new state embedding. The `rng_key` argument is consumed.
    parent_index: the index of the parent node, from which the action will be
      expanded. Shape `[B]`.
    action: the action to expand. Shape `[B]`.
    next_node_index: the index of the newly expanded node. This can be the index
      of an existing node, if `max_depth` is reached. Shape `[B]`.

  Returns:
    tree: updated MCTS tree state.
  """
  batch_size = tree_lib.infer_batch_size(tree)
  batch_range = jnp.arange(batch_size)
  chex.assert_shape([parent_index, action, next_node_index], (batch_size,))

  # Retrieve states for nodes to be evaluated.
  embedding = jax.tree_util.tree_map(
      lambda x: x[batch_range, parent_index], tree.embeddings)

  # Evaluate and create a new node.
  step, embedding = recurrent_fn(params, rng_key, action, embedding)
  chex.assert_shape(step.prior_logits, [batch_size, tree.num_actions])
  chex.assert_shape(step.reward, [batch_size])
  chex.assert_shape(step.discount, [batch_size])
  chex.assert_shape(step.value, [batch_size])
  chex.assert_shape(step.legal_action_mask, [batch_size, tree.num_actions])
  tree = update_tree_node(
      tree, next_node_index, step.prior_logits, step.value, embedding)

  # Return updated tree topology.
  return tree.replace(
      children_index=batch_update(
          tree.children_index, next_node_index, parent_index, action),
      children_rewards=batch_update(
          tree.children_rewards, step.reward, parent_index, action),
      children_discounts=batch_update(
          tree.children_discounts, step.discount, parent_index, action),
      parents=batch_update(tree.parents, parent_index, next_node_index),
      action_from_parent=batch_update(
          tree.action_from_parent, action, next_node_index),
      legal_action_mask=batch_update(
        tree.legal_action_mask, step.legal_action_mask, next_node_index))


@functools.partial(jax.vmap, in_axes=[0, 0, None, None])
def backward(
    tree: Tree[T],
    leaf_index: chex.Numeric,
    backward_method: str,
    alpha: float) -> Tree[T]:
  """Goes up and updates the tree until all nodes reached the root.

  Args:
    tree: the MCTS tree state to update, without the batch size.
    leaf_index: the node index from which to do the backward.

  Returns:
    Updated MCTS tree state.
  """

  def cond_fun(loop_state):
    _, _, index = loop_state
    return index != Tree.ROOT_INDEX

  def body_fun(loop_state):
    # Here we update the value of our parent, so we start by reversing.
    tree, leaf_value, index = loop_state
    parent = tree.parents[index]
    count = tree.node_visits[parent]
    action = tree.action_from_parent[index]
    reward = tree.children_rewards[parent, action]
    leaf_value = reward + tree.children_discounts[parent, action] * leaf_value
    parent_value = (
        tree.node_values[parent] * count + leaf_value) / (count + 1.0)
    children_values = tree.node_values[index]
    children_counts = tree.children_visits[parent, action] + 1

    tree = tree.replace(
        node_values=update(tree.node_values, parent_value, parent),
        node_visits=update(tree.node_visits, count + 1, parent),
        children_values=update(
            tree.children_values, children_values, parent, action),
        children_visits=update(
            tree.children_visits, children_counts, parent, action))

    return tree, leaf_value, parent
  
  def afn_body_fun(loop_state):
    """
    Compute recursive updates using the generalised TB update (also known as
    the mellowmax operator).

    NOTE: `leaf_value` is here used to represent the flow of the prior node in\
          the search tree. The leaf value is expected to be from the\
          perspective of the parent. Thus, for each node, we must invert its\
          flow prediction to put it in the perspective of the opponent.
    """
    # Here we update the value of our parent, so we start by reversing.
    tree, leaf_value, index = loop_state
    parent = tree.parents[index]
    count = tree.node_visits[parent]
    action = tree.action_from_parent[index]
    reward = tree.children_rewards[parent, action]
    leaf_value = reward + tree.children_discounts[parent, action] * leaf_value

    # Use the current tree.children_values as the flow estimates, and
    # therefore, as the current estimate of the QF values. Mask out the invalid
    # actions to prevent log-sum-exp errors.
    prior_values = jnp.where(
      tree.legal_action_mask[parent],
      tree.children_values[parent],
      jnp.finfo(tree.children_values.dtype).min,
    )

    # TODO TESTING: Setting prior_logits[parent, action] = leaf_value.
    prior_values = prior_values.at[action].set(leaf_value)
    new_parent_value = -(
      leaf_value - prior_values[action] +
      jsp.special.logsumexp((alpha + 1) * prior_values) -
      jsp.special.logsumexp(alpha * prior_values)
    )

    # Take the average parent value with this update since we don't want a
    # single bad estimate to throw off the entire estimate.
    parent_value = (
        tree.node_values[parent] * count + new_parent_value) / (count + 1.0)
    
    # Set up the updates for the various tracked values. NOTE: we must ensure
    # that node_values and children_values are zeroed out for terminal and
    # post-terminal states!!! The value of a terminal state must come only from
    # the reward obtained at that state!
    discount = tree.children_discounts[parent, action]
    leaf_value = parent_value # Can't do anything here, discount is 1...
    new_child_value = discount * tree.node_values[index]
    new_child_prior_logits = discount * tree.node_values[index]
    children_counts = tree.children_visits[parent, action] + 1

    tree = tree.replace(
        node_values=update(tree.node_values, parent_value, parent),
        node_visits=update(tree.node_visits, count + 1, parent),
        children_prior_logits=update(tree.children_prior_logits,
            new_child_prior_logits, parent, action),
        children_values=update(
            tree.children_values, new_child_value, parent, action),
        children_visits=update(
            tree.children_visits, children_counts, parent, action))

    return tree, leaf_value, parent
  
  def afn_const_body_fun(loop_state):
    """
    Compute recursive updates using the generalised TB update (also known as
    the mellowmax operator). This code also predicts a constant from the
    softmax function.

    NOTE: This is almost the exact same as the afn_body_fun, but instead of\
          trusting the QF values since they are likely trained via a KL\
          divergence on a softmax-based policy, we need to predict the\
          constant of the softmax. As such, this backward function also\
          predicts the constant and uses it to correct the flow prediction.
    """
    # Here we update the value of our parent, so we start by reversing.
    tree, leaf_value, index = loop_state
    parent = tree.parents[index]
    count = tree.node_visits[parent]
    action = tree.action_from_parent[index]
    reward = tree.children_rewards[parent, action]
    leaf_value = reward + tree.children_discounts[parent, action] * leaf_value
    child_consts = tree.children_QF_const[index]
    child_visits = tree.children_visits[index]
    sqrt_child_visits = jnp.sqrt(child_visits)

    # Estimate the constant from the children's constants. Two methods here
    # again: 1) simple average, 2) Kolya's sqrt(N)-weighted average.
    child_sum = jnp.sum(child_visits)
    child_root_sum = jnp.sum(sqrt_child_visits)
    # If the child visitation sum is zero, set the constant to zero.
    """ 
    NOTE: If the sum is zero, this indicates that we have no information on the
          constant anyways: we are either at a terminal node, indicating that
          we have achieved the true, unbiased reward, or we are at a leaf node,
          meaning that we don't have any flow estimates from lower in the tree
          to actually construct the constant estimate. So, we have to trust the
          flow estimate at this point.
    """
    curr_const_est_1 = jnp.select(
      condlist=[child_sum == 0.0],
      choicelist=[0.0],
      default=jnp.sum(child_consts) / child_sum,
    )
    curr_const_est_2 = jnp.select(
      condlist=[child_root_sum == 0.0],
      choicelist=[0.0],
      default=jnp.sum(child_consts * sqrt_child_visits) / child_root_sum,
    )
    
    # Use the current tree.children_values as the flow estimates, and
    # therefore, as the current estimate of the QF values. Mask out the invalid
    # actions to prevent log-sum-exp errors.
    prior_values = jnp.where(
      tree.legal_action_mask[parent],
      tree.children_values[parent],
      jnp.finfo(tree.children_values.dtype).min,
    )

    # TODO: TESTING Setting prior_logits[parent, action] = leaf_value. This has
    #       a very similar update structure to the MENTS paper. Also shows
    #       slight reduction in error compared to not doing it this way.
    prior_values = prior_values.at[action].set(leaf_value)
    pv_adjusted = prior_values - curr_const_est_1 # Adjust the QF value.
    new_parent_value = -(
      leaf_value - prior_values[action] +
      jsp.special.logsumexp((alpha + 1) * prior_values) -
      jsp.special.logsumexp(alpha * prior_values)
    )

    # Take the average parent value with this update since we don't want a
    # single bad estimate to throw off the entire estimate.
    parent_value = (
        tree.node_values[parent] * count + new_parent_value) / (count + 1.0)
    
    # Estimate the QF constant on the new flow estimate.
    """
    NOTE: There are a ton of ways to estimate the constant on QF. The problem
          with a tree search scenario is that in the current function, we only
          have access to the current node, not the nodes that come after the
          current node. As such, we are restricted to estimating the constant
          and propagating it up to the next level of the tree search.
    """
    QF_const_est = tree.children_prior_logits[parent, action] - parent_value
    c_count = tree.children_visits[parent, action]
    parent_QF_const = (tree.children_QF_const[parent, action] * c_count +
                       QF_const_est) / (c_count + 1)

    # Set up the updates for the various tracked values. NOTE: we must ensure
    # that node_values and children_values are zeroed out for terminal and
    # post-terminal states!!! The value of a terminal state must come only from
    # the reward obtained at that state!
    discount = tree.children_discounts[parent, action]
    leaf_value = parent_value # Can't do anything here, discount is 1...
    new_child_value = discount * tree.node_values[index]
    new_child_prior_logits = discount * tree.node_values[index]
    children_counts = tree.children_visits[parent, action] + 1

    tree = tree.replace(
        node_values=update(tree.node_values, parent_value, parent),
        node_visits=update(tree.node_visits, count + 1, parent),
        children_QF_const=update(tree.children_QF_const, parent_QF_const,
            parent, action),
        children_values=update(
            tree.children_values, new_child_value, parent, action),
        children_visits=update(
            tree.children_visits, children_counts, parent, action))

    return tree, leaf_value, parent

  if(backward_method == "AFN"):
    body_method = afn_body_fun
  elif(backward_method == "AFN_CONST"):
    body_method = afn_const_body_fun
  else:
    body_method = body_fun
  leaf_index = jnp.asarray(leaf_index, dtype=jnp.int32)
  loop_state = (tree, tree.node_values[leaf_index], leaf_index)
  tree, _, _ = jax.lax.while_loop(cond_fun, body_method, loop_state)

  return tree


# Utility function to set the values of certain indices to prescribed values.
# This is vmapped to operate seamlessly on batches.
def update(x, vals, *indices):
  return x.at[indices].set(vals)


batch_update = jax.vmap(update)


def update_tree_node(
    tree: Tree[T],
    node_index: chex.Array,
    prior_logits: chex.Array,
    value: chex.Array,
    embedding: chex.Array) -> Tree[T]:
  """Updates the tree at node index.

  Args:
    tree: `Tree` to whose node is to be updated.
    node_index: the index of the expanded node. Shape `[B]`.
    prior_logits: the prior logits to fill in for the new node, of shape
      `[B, num_actions]`.
    value: the value to fill in for the new node. Shape `[B]`.
    embedding: the state embeddings for the node. Shape `[B, ...]`.

  Returns:
    The new tree with updated nodes.
  """
  batch_size = tree_lib.infer_batch_size(tree)
  batch_range = jnp.arange(batch_size)
  chex.assert_shape(prior_logits, (batch_size, tree.num_actions))

  # When using max_depth, a leaf can be expanded multiple times.
  new_visit = tree.node_visits[batch_range, node_index] + 1
  # parent = tree.parents[batch_range, node_index]
  # action = tree.action_from_parent[batch_range, parent]
  # new_child_visit = tree.children_visits[batch_range, parent, action] + 1
  updates = dict(  # pylint: disable=use-dict-literal
      children_prior_logits=batch_update(
          tree.children_prior_logits, prior_logits, node_index),
      raw_values=batch_update(
          tree.raw_values, value, node_index),
      node_values=batch_update(
          tree.node_values, value, node_index),
      node_visits=batch_update(
          tree.node_visits, new_visit, node_index),
      # children_visits=batch_update(
      #     tree.children_visits, new_child_visit, parent, action),
      embeddings=jax.tree_util.tree_map(
          lambda t, s: batch_update(t, s, node_index),
          tree.embeddings, embedding))

  return tree.replace(**updates)


def instantiate_tree_from_root(
    root: base.RootFnOutput,
    num_simulations: int,
    root_invalid_actions: chex.Array,
    extra_data: Any) -> Tree:
  """Initializes tree state at search root."""
  chex.assert_rank(root.prior_logits, 2)
  batch_size, num_actions = root.prior_logits.shape
  chex.assert_shape(root.value, [batch_size])
  num_nodes = num_simulations + 1
  data_dtype = root.value.dtype
  min_val = jnp.finfo(data_dtype).min
  batch_node = (batch_size, num_nodes)
  batch_node_action = (batch_size, num_nodes, num_actions)

  def _zeros(x):
    return jnp.zeros(batch_node + x.shape[1:], dtype=x.dtype)
  
  # Create a new empty tree state and fill its root.
  tree = Tree(
      node_visits=jnp.zeros(batch_node, dtype=jnp.int32),
      raw_values=jnp.zeros(batch_node, dtype=data_dtype),
      node_values=jnp.zeros(batch_node, dtype=data_dtype),
      parents=jnp.full(batch_node, Tree.NO_PARENT, dtype=jnp.int32),
      action_from_parent=jnp.full(
          batch_node, Tree.NO_PARENT, dtype=jnp.int32),
      children_index=jnp.full(
          batch_node_action, Tree.UNVISITED, dtype=jnp.int32),
      children_prior_logits=jnp.zeros(
          batch_node_action, dtype=root.prior_logits.dtype),
      children_values=min_val * jnp.ones(batch_node_action, dtype=data_dtype),
      children_visits=jnp.zeros(batch_node_action, dtype=jnp.int32),
      children_rewards=jnp.zeros(batch_node_action, dtype=data_dtype),
      children_discounts=jnp.zeros(batch_node_action, dtype=data_dtype),
      children_QF_const=jnp.zeros(batch_node_action, dtype=data_dtype),
      embeddings=jax.tree_util.tree_map(_zeros, root.embedding),
      root_invalid_actions=root_invalid_actions,
      legal_action_mask=jnp.zeros(batch_node_action, dtype=jnp.bool),
      extra_data=extra_data)

  tree = tree.replace(
    legal_action_mask = tree.legal_action_mask.at[:, tree.ROOT_INDEX].set(
      ~root_invalid_actions))

  root_index = jnp.full([batch_size], Tree.ROOT_INDEX)
  tree = update_tree_node(
      tree, root_index, root.prior_logits, root.value, root.embedding)
  # from mctx._src.search import instantiate_tree_from_root as inst_tree; root_out = mctx.RootFnOutput(prior_logits=jnp.array([[1, 2]]), value=jnp.array([10]), embedding=None); new_tree = inst_tree(root_out, 50, jnp.array([[0, 0]]), None);
  return tree
